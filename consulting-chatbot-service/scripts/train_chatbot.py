#!/usr/bin/env python
"""
Main Training Script cho AI Chatbot
Run: python scripts/train_chatbot.py

Steps:
1. Generate training data
2. Build tokenizer
3. Create model
4. Train
5. Save model
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from datetime import datetime

from app.dl_models.config import ModelConfig, SMALL_CONFIG, DEFAULT_CONFIG
from app.dl_models.tokenizer import VietnameseTokenizer, create_tokenizer_from_knowledge_base
from app.dl_models.chatbot_model import ChatbotModel
from app.training.data_generator import ChatbotDataGenerator, generate_training_data
from app.training.dataset import load_samples_from_json, create_dataloaders
from app.training.trainer import ChatbotTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train AI Chatbot')
    
    parser.add_argument('--config', type=str, default='default',
                        choices=['small', 'default', 'large'],
                        help='Model configuration size')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to existing training data JSON')
    
    parser.add_argument('--kb-path', type=str,
                        default='../knowledge-base/documents',
                        help='Path to knowledge base documents')
    
    parser.add_argument('--save-dir', type=str, default='saved_models',
                        help='Directory to save models')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("AI Chatbot Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_data_dir = project_root / "training_data"
    training_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate or load training data
    print("\n" + "=" * 60)
    print("Step 1: Preparing Training Data")
    print("=" * 60)
    
    data_path = args.data_path
    if data_path is None:
        data_path = training_data_dir / "chatbot_data.json"
        
        if not data_path.exists():
            print("Generating training data...")
            generator = ChatbotDataGenerator()
            generator.generate_all(
                greeting_n=5,
                product_n=300,
                policy_n=5,
                order_n=150,
                recommendation_n=150,
                general_n=5
            )
            generator.save(str(data_path))
            
            print("\nIntent distribution:")
            for intent, count in generator.get_intent_distribution().items():
                print(f"  {intent}: {count}")
        else:
            print(f"Using existing data: {data_path}")
    
    # Load samples
    samples = load_samples_from_json(str(data_path))
    print(f"Loaded {len(samples)} samples")
    
    # Step 2: Build tokenizer
    print("\n" + "=" * 60)
    print("Step 2: Building Tokenizer")
    print("=" * 60)
    
    tokenizer_path = save_dir / "tokenizer.pkl"
    
    if tokenizer_path.exists() and args.resume:
        print(f"Loading existing tokenizer: {tokenizer_path}")
        tokenizer = VietnameseTokenizer.load(str(tokenizer_path))
    else:
        # Build from training data + knowledge base
        all_texts = [s.query for s in samples] + [s.response for s in samples]
        
        # Add knowledge base if available
        kb_path = Path(args.kb_path)
        if kb_path.exists():
            print(f"Including knowledge base: {kb_path}")
            for md_file in kb_path.rglob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_texts.extend([
                        line.strip() for line in content.split('\n')
                        if line.strip()
                    ])
        
        print(f"Building tokenizer from {len(all_texts)} texts...")
        tokenizer = VietnameseTokenizer(vocab_size=10000, max_length=128)
        tokenizer.fit(all_texts)
        tokenizer.save(str(tokenizer_path))
    
    print(f"Vocabulary size: {tokenizer.vocab_len}")
    
    # Step 3: Create dataloaders
    print("\n" + "=" * 60)
    print("Step 3: Creating DataLoaders")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        samples=samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        max_input_length=64,
        max_target_length=128
    )
    
    # Step 4: Create model
    print("\n" + "=" * 60)
    print("Step 4: Creating Model")
    print("=" * 60)
    
    config_map = {
        'small': SMALL_CONFIG,
        'default': DEFAULT_CONFIG,
        'large': ModelConfig(
            vocab_size=tokenizer.vocab_len,
            embedding_dim=512,
            encoder_hidden_size=512,
            decoder_hidden_size=1024
        )
    }
    
    config = config_map[args.config]
    config.vocab_size = tokenizer.vocab_len
    
    model = ChatbotModel(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")
    print(f"Config: {config.to_dict()}")
    
    # Step 5: Train
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)
    
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        teacher_forcing_ratio=0.5,
        patience=7
    )
    
    trainer = ChatbotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        save_dir=str(save_dir),
        device=args.device
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    history = trainer.train()
    
    # Step 6: Final evaluation
    print("\n" + "=" * 60)
    print("Step 6: Final Evaluation")
    print("=" * 60)
    
    # Load best model
    model = ChatbotModel.load(str(save_dir / "best_model.pt"), trainer.device)
    model.eval()
    
    # Test on some examples
    test_queries = [
        "Xin chào",
        "Tìm sách về kinh doanh",
        "Chính sách đổi trả như thế nào?",
        "Đơn hàng BK-123456 của tôi ở đâu?",
        "Gợi ý sách cho tôi",
        "Cảm ơn bạn"
    ]
    
    print("\nTest predictions:")
    for query in test_queries:
        input_ids, length = tokenizer.encode(query, max_length=64, return_length=True)
        input_tensor = torch.tensor([input_ids], device=trainer.device)
        lengths = torch.tensor([length], device=trainer.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_tensor,
                lengths=lengths,
                max_length=50,
                temperature=0.8
            )
        
        response_ids = output['sequences'][0].cpu().tolist()
        response = tokenizer.decode(response_ids)
        intent_idx = output['intent'][0].cpu().item()
        intents = ["greeting", "product_query", "policy_query", 
                   "order_support", "recommendation", "general_chat"]
        
        print(f"\nQ: {query}")
        print(f"Intent: {intents[intent_idx]}")
        print(f"A: {response[:200]}...")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Models saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
