import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ShoppingCart, Search, User, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      navigate(`/products?search=${encodeURIComponent(searchTerm)}`);
    }
  };

  return (
    <header className="header-wrapper">
      <div className="header-main">
        <div className="navbar-brand">
          <Link to="/">
             Nova<span>Store</span>
          </Link>
        </div>
        
        <form className="search-bar-container" onSubmit={handleSearch}>
          <input 
            type="text" 
            placeholder="Search for modern products..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button type="submit" className="search-btn"><Search size={18} /></button>
        </form>

        <div className="header-actions">
          {user ? (
            <>
              <div className="header-link">
                 <User size={20} />
                 <span>{user.email.split('@')[0]}</span>
              </div>
              <Link to="/orders" className="header-link">
                 <span>My Orders</span>
              </Link>
              <Link to="/cart" className="header-cart">
                <ShoppingCart size={24} />
                <span className="cart-badge">2</span>
              </Link>
              <button onClick={handleLogout} className="btn-outline" style={{padding: '0.5rem 1rem'}}>
                 <LogOut size={16} /> Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn-outline" style={{padding: '0.5rem 1.5rem'}}>Sign In</Link>
              <Link to="/login" className="btn-primary">Register</Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
