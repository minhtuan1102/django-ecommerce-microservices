import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ShoppingCart } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import api from '../api';

interface Product {
  id: number;
  name: string;
  price: string | number;
  stock: number;
  category_name?: string;
  metadata?: string;
}

export const ProductDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [product, setProduct] = useState<Product | null>(null);
  const [quantity, setQuantity] = useState(1);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Product[]>([]);
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProduct = async () => {
      try {
        const res = await api.get(`/products/${id}/`);
        setProduct(res.data);
      } catch (err) {
        console.error('Failed to fetch product', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProduct();

    const fetchRecs = async () => {
      try {
        const res = await api.get(`/recommend?user_id=${user?.id || 1}&product_id=${id}`);
        if (Array.isArray(res.data)) {
            // Mock fetching the real products for these IDs since backend only returns IDs
            const recsPromises = res.data.slice(0, 4).map(recId => 
               api.get(`/products/${recId}/`).then(r => r.data).catch(() => null)
            );
            const resolvedRecs = await Promise.all(recsPromises);
            setRecommendations(resolvedRecs.filter(r => r !== null));
        }
      } catch (err) {
        console.error('Failed to fetch recommendations', err);
      }
    };
    fetchRecs();
  }, [id, user]);

  const addToCart = async (buyNow = false) => {
    if (!user) {
      navigate('/login');
      return;
    }
    try {
      await api.post('/cart/cart-items/', {
        customer_id: user.id,
        product_id: parseInt(id!),
        quantity: quantity
      });
      if (buyNow) {
          navigate('/cart');
      } else {
          alert('Item has been added to your shopping cart');
      }
    } catch (err) {
      console.error('Failed to add to cart', err);
      alert('Failed to add to cart');
    }
  };

  const handleQtyChange = (val: number) => {
    if (!product) return;
    let newVal = quantity + val;
    if (newVal < 1) newVal = 1;
    if (newVal > product.stock) newVal = product.stock;
    setQuantity(newVal);
  }

  if (loading) return <div className="loader">Loading product details...</div>;
  if (!product) return <div style={{textAlign: 'center', padding: '50px'}}>Product not found.</div>;

  const currentPrice = Number(product.price);
  const oldPrice = currentPrice * 1.5;

  return (
    <div className="product-detail-container">
      <div className="product-detail-card">
        <div className="product-image-gallery">
          <div className="product-image-placeholder large">
            <img src={`https://picsum.photos/seed/${product.id}/450/450`} alt={product.name} style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
          </div>
        </div>
        
        <div className="product-info">
          <h2>{product.name}</h2>
          
          <div className="product-stats">
              <div><span className="rating">4.9</span> <span className="stars">★★★★★</span></div>
              <div style={{borderLeft: '1px solid #ccc', paddingLeft: '20px'}}><span>1.2k</span> Ratings</div>
              <div style={{borderLeft: '1px solid #ccc', paddingLeft: '20px'}}><span>3.5k</span> Sold</div>
          </div>

          <div className="price-block">
             <span className="old-price">₫{oldPrice.toLocaleString('vi-VN')}</span>
             <span className="current-price">₫{currentPrice.toLocaleString('vi-VN')}</span>
          </div>
          
          <div className="attribute-row">
             <div className="label">Category</div>
             <div>{product.category_name || 'General'}</div>
          </div>

          <div className="attribute-row">
             <div className="label">Shipping</div>
             <div>Free Shipping</div>
          </div>

          <div className="attribute-row">
            <div className="label">Quantity</div>
            <div style={{display: 'flex', alignItems: 'center'}}>
                <div className="quantity-selector">
                    <button onClick={() => handleQtyChange(-1)} disabled={quantity <= 1}>-</button>
                    <input 
                        type="text" 
                        value={quantity} 
                        readOnly
                    />
                    <button onClick={() => handleQtyChange(1)} disabled={quantity >= product.stock}>+</button>
                </div>
                <span className="available-stock">{product.stock} pieces available</span>
            </div>
          </div>

          <div className="action-buttons">
            <button onClick={() => addToCart(false)} className="btn-outline" disabled={product.stock === 0}>
               <ShoppingCart size={20}/> Add To Cart
            </button>
            <button onClick={() => addToCart(true)} className="btn-primary" disabled={product.stock === 0}>
               Buy Now
            </button>
          </div>
        </div>
      </div>
      
      {recommendations.length > 0 && (
        <div className="recommendations-section">
          <div className="section-title">You may also like</div>
          <div className="product-grid">
            {recommendations.map(rec => (
              <Link key={rec.id} to={`/products/${rec.id}`} className="product-card">
                <div className="product-image-placeholder">
                  <img src={`https://picsum.photos/seed/${rec.id}/200/200`} alt={rec.name} style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
                </div>
                <div className="product-content">
                  <h3>{rec.name}</h3>
                  <div className="product-price-row">
                    <span className="price">{Number(rec.price).toLocaleString('vi-VN')}</span>
                    <span className="sold">{Math.floor(Math.random() * 5000)} sold</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
