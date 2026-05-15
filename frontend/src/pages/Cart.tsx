import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api';

interface CartItem {
  id: number;
  book_id: number;
  quantity: number;
  product?: { name: string, price: string | number, metadata?: { image_url?: string } };
}

export const Cart = () => {
  const [items, setItems] = useState<CartItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [checkoutMode, setCheckoutMode] = useState(false);
  const [address, setAddress] = useState("");
  const [phone, setPhone] = useState("");
  const [paymentMethod, setPaymentMethod] = useState("cod");
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }
    fetchCart();
  }, [user, navigate]);

  const [cartId, setCartId] = useState<number | null>(null);

  const fetchCart = async () => {
    try {
      const res = await api.get(`/cart/carts/${user?.id}/`);
      const cartItems = res.data.items || [];
      const cartId = res.data.cart_id;
      
      // Fetch product details for each item
      const detailedItems = await Promise.all(cartItems.map(async (item: CartItem) => {
        try {
          const productRes = await api.get(`/products/${item.book_id}/`);
          return { ...item, product: productRes.data };
        } catch (e) {
          console.error(`Failed to fetch product ${item.book_id}`, e);
          return item;
        }
      }));

      setItems(detailedItems);
      setCartId(cartId);
    } catch (err) {
      console.error('Failed to fetch cart', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckout = async () => {
    if (!address.trim()) {
      alert("Please enter your shipping address");
      return;
    }
    if (!phone.trim()) {
      alert("Please enter your phone number");
      return;
    }
    setLoading(true);
    try {
      const orderRes = await api.post('/orders/', {
        customer_id: user?.id,
        items: items.map(item => ({ book_id: item.book_id, quantity: item.quantity, price: Number(item.product?.price || 0) })),
        total_price: calculateTotal(),
        shipping_fee: 0,
        shipping_address: `${address} (Phone: ${phone})`,
        payment_method: paymentMethod
      });
      const orderId = orderRes.data.id;

      // Clear cart on backend after checkout
      await api.delete(`/cart/carts/${user?.id}/clear/`);

      alert('Checkout successful! Flow: Order -> Payment -> Shipping');
      navigate('/orders');
    } catch (err: any) {
      console.error('Checkout failed', err);
      const errorDetail = err.response?.data?.detail 
        || JSON.stringify(err.response?.data) 
        || err.message;
      alert(`Checkout process failed: ${errorDetail}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (productId: number) => {
      try {
          await api.delete(`/cart/cart-items/${cartId}/${productId}/`);
          fetchCart();
      } catch (err) {
          console.error('Failed to remove item', err);
      }
  };

  const calculateTotal = () => {
    return items.reduce((acc, item) => {
      const price = item.product ? Number(item.product.price) : 0;
      return acc + (item.quantity * price);
    }, 0);
  };

  if (loading) return <div className="loader">Loading cart...</div>;

  return (
    <div className="cart-container">
      {items.length === 0 ? (
        <div style={{textAlign: 'center', padding: '100px 0', background: 'white'}}>
           <img src="https://deo.shopeemobile.com/shopee/shopee-pcmall-live-sg/cart/9bdd8040b334d31946f49e36beaf32db.png" alt="Empty Cart" style={{width: '100px', marginBottom: '20px'}}/>
           <p style={{color: '#888', fontWeight: 500, fontSize: '14px', marginBottom: '20px'}}>Your shopping cart is empty</p>
           <button onClick={() => navigate('/')} className="btn-primary" style={{padding: '10px 40px'}}>Go Shopping Now</button>
        </div>
      ) : (
        <div className="cart-content" style={{padding: 0, boxShadow: 'none', background: 'transparent'}}>
          {checkoutMode ? (
            <div className="checkout-layout" style={{display: 'flex', gap: '30px', alignItems: 'flex-start', color: '#1f2937'}}>
              <div className="checkout-form" style={{flex: 2, background: '#ffffff', padding: '30px', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', color: '#1f2937'}}>
                <h2 style={{borderBottom: '1px solid #ebebeb', paddingBottom: '15px', marginBottom: '20px', color: '#0f172a'}}>Checkout Information</h2>
                
                <div style={{display: 'flex', gap: '15px', marginBottom: '20px'}}>
                  <div style={{flex: 1}}>
                    <label style={{display: 'block', marginBottom: '8px', fontWeight: 'bold', fontSize: '14px', color: '#111827'}}>Phone Number:</label>
                    <input 
                      type="text"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      placeholder="e.g. 0912345678"
                      style={{width: '100%', padding: '10px', border: '1px solid #d1d5db', borderRadius: '5px', boxSizing: 'border-box', background: '#ffffff', color: '#111827'}}
                    />
                  </div>
                </div>

                <div style={{marginBottom: '20px'}}>
                  <label style={{display: 'block', marginBottom: '8px', fontWeight: 'bold', fontSize: '14px', color: '#111827'}}>Shipping Address:</label>
                  <textarea 
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    placeholder="Enter your detailed address (Home, Ward, District, City)..."
                    style={{width: '100%', padding: '10px', height: '80px', border: '1px solid #d1d5db', borderRadius: '5px', boxSizing: 'border-box', resize: 'vertical', background: '#ffffff', color: '#111827'}}
                  />
                </div>

                <div style={{marginBottom: '25px'}}>
                  <label style={{display: 'block', marginBottom: '10px', fontWeight: 'bold', fontSize: '14px', color: '#111827'}}>Payment Method:</label>
                  <div style={{display: 'flex', gap: '20px'}}>
                    <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '10px 15px', border: paymentMethod === 'cod' ? '2px solid #ee4d2d' : '1px solid #d1d5db', borderRadius: '5px', background: paymentMethod === 'cod' ? '#fff6f5' : '#ffffff', color: '#111827'}}>
                      <input type="radio" value="cod" checked={paymentMethod === 'cod'} onChange={(e) => setPaymentMethod(e.target.value)} style={{display: 'none'}} />
                      🚚 Cash on Delivery (COD)
                    </label>
                    <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '10px 15px', border: paymentMethod === 'credit_card' ? '2px solid #ee4d2d' : '1px solid #d1d5db', borderRadius: '5px', background: paymentMethod === 'credit_card' ? '#fff6f5' : '#ffffff', color: '#111827'}}>
                      <input type="radio" value="credit_card" checked={paymentMethod === 'credit_card'} onChange={(e) => setPaymentMethod(e.target.value)} style={{display: 'none'}} />
                      💳 Credit/Debit Card
                    </label>
                  </div>
                </div>

                <div style={{display: 'flex', gap: '15px', justifyContent: 'flex-start', marginTop: '30px'}}>
                   <button onClick={() => setCheckoutMode(false)} className="btn-secondary" style={{padding: '12px 25px', background: '#f8fafc', color: '#111827', border: '1px solid #d1d5db', cursor: 'pointer', borderRadius: '5px', fontWeight: 'bold'}}>Back to Cart</button>
                   <button onClick={handleCheckout} className="btn-primary checkout-btn" style={{padding: '12px 30px', margin: 0, fontWeight: 'bold', background: '#ee4d2d', color: '#ffffff', border: 'none'}}>Place Order</button>
                </div>
              </div>

              <div className="checkout-summary" style={{flex: 1, background: '#ffffff', padding: '25px', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', color: '#1f2937'}}>
                <h3 style={{marginBottom: '20px', fontSize: '18px', color: '#0f172a'}}>Order Summary</h3>
                <div style={{maxHeight: '300px', overflowY: 'auto', marginBottom: '20px', paddingRight: '5px'}}>
                  {items.map(item => (
                    <div key={item.id} style={{display: 'flex', gap: '10px', marginBottom: '15px'}}>
                      <img src={item.product?.metadata?.image_url || `https://picsum.photos/seed/${item.book_id}/50/50`} alt="Product" style={{width: '50px', height: '50px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #eee'}}/>
                      <div style={{flex: 1}}>
                        <div style={{fontSize: '13px', fontWeight: 'bold', color: '#111827', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden'}}>{item.product?.name || `Product #${item.book_id}`}</div>
                        <div style={{fontSize: '12px', color: '#6b7280', marginTop: '4px'}}>Qty: {item.quantity}</div>
                      </div>
                      <div style={{fontSize: '13px', fontWeight: 'bold', color: '#ee4d2d'}}>
                        ₫{(item.quantity * Number(item.product?.price || 0)).toLocaleString('vi-VN')}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div style={{borderTop: '1px dashed #e5e7eb', paddingTop: '15px', marginBottom: '10px', display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#374151'}}>
                  <span>Subtotal:</span>
                  <span>₫{calculateTotal().toLocaleString('vi-VN')}</span>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#374151', marginBottom: '15px'}}>
                  <span>Shipping Fee:</span>
                  <span>₫0</span>
                </div>
                <div style={{borderTop: '1px solid #e5e7eb', paddingTop: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <span style={{fontSize: '16px', fontWeight: 'bold', color: '#0f172a'}}>Total:</span>
                  <span style={{fontSize: '22px', fontWeight: 'bold', color: '#ee4d2d'}}>₫{calculateTotal().toLocaleString('vi-VN')}</span>
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="cart-header-table">
                 <div>Product</div>
                 <div>Unit Price</div>
                 <div>Quantity</div>
                 <div>Total Price</div>
                 <div>Actions</div>
              </div>
              
              <div className="cart-list">
                {items.map((item) => (
                  <div key={item.id} className="cart-item-row">
                    <div className="cart-product-info">
                       <div className="cart-product-img">
                           <img src={item.product?.metadata?.image_url || `https://picsum.photos/seed/${item.book_id}/80/80`} alt="Product" style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
                       </div>
                       <div className="cart-product-name">{item.product?.name || `Product ID: #${item.book_id}`}</div>
                    </div>
                    <div className="cart-item-price">₫{(Number(item.product?.price || 0)).toLocaleString('vi-VN')}</div>
                    <div>{item.quantity}</div>
                    <div className="cart-item-total">₫{(item.quantity * Number(item.product?.price || 0)).toLocaleString('vi-VN')}</div>
                    <div>
                      <button onClick={() => handleRemove(item.book_id)} style={{background: 'none', border: 'none', color: '#dc3545', cursor: 'pointer', fontWeight: 'bold'}}>Delete</button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="cart-footer">
                <span className="total-text">Total ({items.length} item{items.length > 1 ? 's' : ''}): </span>
                <span className="total-amount" style={{marginRight: '20px'}}>₫{calculateTotal().toLocaleString('vi-VN')}</span>
                <button onClick={() => setCheckoutMode(true)} className="btn-primary checkout-btn">
                  Check Out
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
