import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import api from '../api';

interface Product {
  id: number;
  name: string;
  price: string | number;
  stock: number;
  category?: { name: string };
}

export const ProductList = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [totalPages, setTotalPages] = useState(1);
  const [searchParams, setSearchParams] = useSearchParams();
  
  const currentPage = parseInt(searchParams.get('page') || '1', 10);

  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true);
      try {
        const res = await api.get(`/products/?page=${currentPage}`);
        setProducts(res.data.products || res.data.results || res.data);
        
        // Assuming typical DRF Pagination returns count and page_size=10
        if (res.data.count) {
           setTotalPages(Math.ceil(res.data.count / 10)); // adjust 10 to your page_size
        } else if (res.data.total_pages) {
           setTotalPages(res.data.total_pages);
        } else if (Array.isArray(res.data) && res.data.length > 0) {
           setTotalPages(1); // No backend pagination
        }
      } catch (err) {
        console.error('Failed to fetch products', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProducts();
  }, [currentPage]);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
       setSearchParams({ page: newPage.toString() });
       window.scrollTo(0, 0);
    }
  };

  if (loading && products.length === 0) return <div className="loader">Loading Daily Discoveries...</div>;

  return (
    <div className="product-container">
      {/* Promotional Banners */}
      <div className="banner-section">
        <div>
           <h2>BIG SALE - Up to 50% Off</h2>
           <p>Discover the best deals on modern essentials today.</p>
        </div>
      </div>

      <div className="section-title">DAILY DISCOVER</div>

      {loading && products.length > 0 && <div style={{textAlign: 'center', marginBottom: '1rem', color: '#64748b'}}>Loading more products...</div>}

      <div className="product-grid">
        {products.map((p) => (
          <Link key={p.id} to={`/products/${p.id}`} className="product-card">
            <div className="product-image-placeholder">
              <img src={`https://picsum.photos/seed/${p.id}/200/200`} alt={p.name}/>
            </div>
            <div className="product-content">
               <h3>{p.name}</h3>
               <div className="product-price-row">
                 <span className="price">₫{Number(p.price).toLocaleString('vi-VN')}</span>
                 <span className="sold">{Math.floor(Math.random() * 5000)} sold</span>
               </div>
            </div>
          </Link>
        ))}
      </div>
      {products.length === 0 && !loading && <p style={{textAlign: 'center', padding: '20px'}}>No products found.</p>}

      {totalPages > 1 && (
        <div className="pagination" style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem', marginTop: '3rem'}}>
           <button 
             onClick={() => handlePageChange(currentPage - 1)} 
             disabled={currentPage === 1}
             className="btn-outline"
             style={{padding: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center'}}
           >
              <ChevronLeft size={20} />
           </button>
           <span style={{fontWeight: 600, color: '#334155'}}>Page {currentPage} of {totalPages}</span>
           <button 
             onClick={() => handlePageChange(currentPage + 1)} 
             disabled={currentPage === totalPages}
             className="btn-outline"
             style={{padding: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center'}}
           >
              <ChevronRight size={20} />
           </button>
        </div>
      )}
    </div>
  );
};
