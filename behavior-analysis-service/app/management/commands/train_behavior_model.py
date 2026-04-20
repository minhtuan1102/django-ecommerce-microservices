from django.core.management.base import BaseCommand

from app.training.train import Trainer


class Command(BaseCommand):
    help = "Train behavior analysis model (real data first, with synthetic fallback)."

    def add_arguments(self, parser):
        parser.add_argument("--customers", type=int, default=1000)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--model-dir", type=str, default="data/models")
        parser.add_argument("--device", type=str, default=None)

        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--use-real-data",
            dest="use_real_data",
            action="store_true",
            help="Prefer real microservice data (default).",
        )
        mode_group.add_argument(
            "--use-synthetic",
            dest="use_real_data",
            action="store_false",
            help="Use synthetic data only.",
        )
        parser.set_defaults(use_real_data=True)

    def handle(self, *args, **options):
        trainer = Trainer(
            model_dir=options["model_dir"],
            n_customers=options["customers"],
            batch_size=options["batch_size"],
            epochs=options["epochs"],
            learning_rate=options["lr"],
            device=options["device"],
            use_synthetic=not options["use_real_data"],
        )

        metrics = trainer.train()
        self.stdout.write(self.style.SUCCESS("Training completed"))
        self.stdout.write(f"Metrics: {metrics}")
