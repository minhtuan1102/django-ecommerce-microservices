import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import api from '../api';
import { Package, Truck, CheckCircle, Clock } from 'lucide-react';

interface OrderItem {
  id: number;
  book_id: number;
  quantity: number;
  price: string;
}

interface Order {
  id: number;
  total_price: string;
  status: string;
  created_at: string;
  items: OrderItem[];
  shipping_status?: string;
  tracking_number?: string;
}

interface OrderItem {
  id: number;
  book_id: number;
  quantity: number;
  price: string;
  product_name?: string; // Sẽ lấy từ product-service
}

interface Order {
  id: number;
  total_price: string;
  shipping_fee: string;
  grand_total: string;
  status: string;
  payment_method: string;
  shipping_address: string;
  created_at: string;
  items: OrderItem[];
  shipping_status?: string;
  tracking_number?: string;
}

export const Orders = () => {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedOrderId, setExpandedOrderId] = useState<number | null>(null);
  const { user } = useAuth();

  useEffect(() => {
    const fetchOrders = async () => {
      try {
        const res = await api.get(`/orders/customer/${user?.id}/`);
        const ordersData: Order[] = res.data;

        // Fetch thêm thông tin vận chuyển và TÊN sản phẩm
        const enrichedOrders = await Promise.all(ordersData.map(async (order) => {
           // 1. Lấy thông tin vận chuyển
           let shippingInfo = {};
           try {
             const shipRes = await api.get(`/shipping/order/${order.id}/`);
             shippingInfo = { shipping_status: shipRes.data.status, tracking_number: shipRes.data.tracking_number };
           } catch (e) {}

           // 2. Lấy tên sản phẩm cho từng item
           const enrichedItems = await Promise.all(order.items.map(async (item) => {
              try {
                const pRes = await api.get(`/products/${item.book_id}/`);
                return { ...item, product_name: pRes.data.name };
              } catch (e) {
                return { ...item, product_name: `Sản phẩm #${item.book_id}` };
              }
           }));

           return { ...order, ...shippingInfo, items: enrichedItems };
        }));

        setOrders(enrichedOrders);
      } catch (err) {
        console.error('Failed to fetch orders', err);
      } finally {
        setLoading(false);
      }
    };
    if (user) fetchOrders();
  }, [user]);

  const toggleOrder = (orderId: number) => {
    setExpandedOrderId(expandedOrderId === orderId ? null : orderId);
  };

  const getPaymentMethodLabel = (method: string) => {
    switch (method) {
      case 'cod': return 'Thanh toán khi nhận hàng (COD)';
      case 'bank_transfer': return 'Chuyển khoản ngân hàng';
      case 'e_wallet': return 'Ví điện tử';
      case 'credit_card': return 'Thẻ tín dụng/Ghi nợ';
      default: return method;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'delivered': return <CheckCircle className="text-accent" />;
      case 'shipping':
      case 'shipped': return <Truck style={{color: '#2563eb'}} />;
      case 'pending': return <Clock style={{color: '#64748b'}} />;
      default: return <Package style={{color: '#64748b'}} />;
    }
  };

  if (loading) return <div className="loader">Đang tải lịch sử đơn hàng...</div>;

  return (
    <div className="orders-container">
      <div className="section-title">Lịch sử mua hàng</div>
      
      {orders.length === 0 ? (
        <div style={{textAlign: 'center', padding: '50px', background: 'white', borderRadius: '8px', border: '1px solid var(--border)'}}>
           <p>Bạn chưa có đơn hàng nào.</p>
        </div>
      ) : (
        <div className="orders-list" style={{display: 'flex', flexDirection: 'column', gap: '1.5rem'}}>
          {orders.map(order => (
            <div 
              key={order.id} 
              className="order-card" 
              onClick={() => toggleOrder(order.id)}
              style={{
                background: 'white', 
                padding: '2rem', 
                borderRadius: '8px', 
                boxShadow: 'var(--shadow-sm)', 
                border: '1px solid var(--border)',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
               {/* Header đơn hàng */}
               <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: expandedOrderId === order.id ? '1.5rem' : '0', borderBottom: expandedOrderId === order.id ? '1px solid #f1f5f9' : 'none', paddingBottom: expandedOrderId === order.id ? '1rem' : '0'}}>
                  <div>
                     <span style={{fontWeight: 700, fontSize: '1.125rem'}}>Mã đơn: #{order.id}</span>
                     <p style={{color: 'var(--text-muted)', fontSize: '0.875rem'}}>{new Date(order.created_at).toLocaleString('vi-VN')}</p>
                  </div>
                  <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                     <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, color: 'var(--primary)'}}>
                        {getStatusIcon(order.shipping_status || order.status)}
                        <span style={{textTransform: 'uppercase'}}>{order.shipping_status || order.status || 'Đang xử lý'}</span>
                     </div>
                     <span style={{fontSize: '0.8rem', color: '#888'}}>{expandedOrderId === order.id ? '▲ Thu gọn' : '▼ Xem chi tiết'}</span>
                  </div>
               </div>

               {/* Chi tiết đơn hàng khi mở rộng */}
               {expandedOrderId === order.id && (
                 <div className="order-details-expanded" style={{marginTop: '1rem'}}>
                    {/* Danh sách sản phẩm */}
                    <div style={{marginBottom: '1.5rem'}}>
                       <h4 style={{fontSize: '1rem', marginBottom: '0.5rem', color: '#334155'}}>Sản phẩm đã mua:</h4>
                       {order.items && order.items.map(item => (
                         <div key={item.id} style={{display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px dashed #f1f5f9'}}>
                           <div style={{flex: 1}}>
                              <div style={{fontWeight: 600}}>{item.product_name}</div>
                              <div style={{fontSize: '0.85rem', color: '#64748b'}}>Số lượng: {item.quantity} x ₫{Number(item.price).toLocaleString('vi-VN')}</div>
                           </div>
                           <div style={{fontWeight: 600, alignSelf: 'center'}}>₫{Number(Number(item.price) * item.quantity).toLocaleString('vi-VN')}</div>
                         </div>
                       ))}
                    </div>

                    {/* Thông tin nhận hàng & Thanh toán */}
                    <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', background: '#f8fafc', padding: '1.5rem', borderRadius: '8px', marginBottom: '1.5rem'}}>
                       <div>
                          <h4 style={{fontSize: '0.9rem', color: '#64748b', marginBottom: '0.5rem', textTransform: 'uppercase'}}>Địa chỉ nhận hàng</h4>
                          <p style={{fontWeight: 500, fontSize: '0.95rem'}}>{order.shipping_address}</p>
                       </div>
                       <div>
                          <h4 style={{fontSize: '0.9rem', color: '#64748b', marginBottom: '0.5rem', textTransform: 'uppercase'}}>Phương thức thanh toán</h4>
                          <p style={{fontWeight: 500, fontSize: '0.95rem'}}>{getPaymentMethodLabel(order.payment_method)}</p>
                       </div>
                    </div>

                    {/* Bảng tính giá */}
                    <div style={{borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem', marginBottom: '1rem'}}>
                       <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.95rem'}}>
                          <span style={{color: '#64748b'}}>Tạm tính:</span>
                          <span>₫{Number(order.total_price).toLocaleString('vi-VN')}</span>
                       </div>
                       <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.95rem'}}>
                          <span style={{color: '#64748b'}}>Phí vận chuyển:</span>
                          <span>₫{Number(order.shipping_fee).toLocaleString('vi-VN')}</span>
                       </div>
                    </div>
                 </div>
               )}

               {/* Footer đơn hàng (luôn hiện tổng tiền) */}
               <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: expandedOrderId === order.id ? '1px solid #f1f5f9' : 'none', paddingTop: expandedOrderId === order.id ? '1rem' : '0.5rem'}}>
                  <div>
                     {order.tracking_number && (
                        <p style={{fontSize: '0.875rem'}}>Mã vận đơn: <span style={{fontFamily: 'monospace', background: '#f1f5f9', padding: '2px 6px'}}>{order.tracking_number}</span></p>
                     )}
                  </div>
                  <div style={{textAlign: 'right'}}>
                     <span style={{color: 'var(--text-muted)', marginRight: '1rem'}}>Tổng thanh toán:</span>
                     <span style={{fontSize: '1.5rem', fontWeight: 800, color: 'var(--secondary)'}}>₫{Number(order.grand_total).toLocaleString('vi-VN')}</span>
                  </div>
               </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
