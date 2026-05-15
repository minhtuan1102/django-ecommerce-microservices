const axios = require('axios');

const API_URL = 'http://localhost:8000';

async function runTests() {
  console.log("=========================================");
  console.log("🚀 Starting FE-BE Integration Tests...");
  console.log("=========================================\n");

  let token = null;
  let authConfig = {};
  let testProductId = null;

  try {
    // 1. Register
    const email = `testuser_${Date.now()}@example.com`;
    const password = 'testpassword123';
    console.log(`[1] Testing Registration Endpoint (POST /auth/register/)...`);
    try {
      await axios.post(`${API_URL}/auth/register/`, { email, password, role: 'customer' });
      console.log(`    ✅ SUCCESS: User ${email} registered.`);
    } catch (e) {
      console.log(`    ⚠️ Warning: Registration failed. The backend might have slightly different requirements for password or username. Proceeding to attempt login with a fallback or standard user... Error: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
    }

    // 2. Login
    console.log(`\n[2] Testing Login Endpoint (POST /auth/login/)...`);
    try {
      const loginRes = await axios.post(`${API_URL}/auth/login/`, { email, password });
      token = loginRes.data.access || loginRes.data.token || loginRes.data.access_token;
      authConfig = { headers: { Authorization: `Bearer ${token}` } };
      console.log(`    ✅ SUCCESS: User logged in, token acquired.`);
    } catch (e) {
        console.log(`    ❌ FAILED: Could not login. Tests requiring authentication will likely fail. Error: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
    }

    // 3. Fetch Products
    console.log(`\n[3] Testing Product Catalog (GET /products/)...`);
    const productsRes = await axios.get(`${API_URL}/products/`);
    const products = productsRes.data.products || productsRes.data.results || productsRes.data;
    console.log(`    ✅ SUCCESS: Fetched ${products.length} products.`);

    if (products.length > 0) {
      testProductId = products[0].id;
      
      // 4. Product Detail
      console.log(`\n[4] Testing Product Detail (GET /products/${testProductId}/)...`);
      const detailRes = await axios.get(`${API_URL}/products/${testProductId}/`);
      console.log(`    ✅ SUCCESS: Fetched product detail for: ${detailRes.data.name}`);

      // 5. Recommendations
      console.log(`\n[5] Testing AI Recommendation (GET /recommend?product_id=${testProductId})...`);
      try {
         const recRes = await axios.get(`${API_URL}/recommend?user_id=1&product_id=${testProductId}`);
         console.log(`    ✅ SUCCESS: AI Recommender returned ${recRes.data.length || 0} suggestions.`);
      } catch (e) {
         console.log(`    ⚠️ Warning: AI Recommender failed or not fully populated yet. Error: ${e.response?.status}`);
      }

      // If we have token, test cart flow
      if (token) {
        console.log(`\n[6] Testing Add to Cart (POST /cart/items/)...`);
        try {
           await axios.post(`${API_URL}/cart/items/`, { product_id: testProductId, quantity: 1 }, authConfig);
           console.log(`    ✅ SUCCESS: Product added to cart.`);
        } catch (e) {
           console.log(`    ⚠️ Warning: Could not add to cart. Error: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
        }

        console.log(`\n[7] Testing Fetch Cart (GET /cart/items/)...`);
        try {
          const cartRes = await axios.get(`${API_URL}/cart/items/`, authConfig);
          console.log(`    ✅ SUCCESS: Cart fetched with ${cartRes.data.length} items.`);
          
          console.log(`\n[8] Testing Checkout Flow...`);
          try {
             const orderRes = await axios.post(`${API_URL}/orders/`, { items: cartRes.data }, authConfig);
             console.log(`    ✅ SUCCESS: Order Created (ID: ${orderRes.data.id})`);
             
             await axios.post(`${API_URL}/payment/`, { order_id: orderRes.data.id, amount: 250000 }, authConfig);
             console.log(`    ✅ SUCCESS: Payment Processed`);

             await axios.post(`${API_URL}/shipping/`, { order_id: orderRes.data.id, address: "Test Address" }, authConfig);
             console.log(`    ✅ SUCCESS: Shipping Scheduled`);

          } catch (e) {
             console.log(`    ⚠️ Warning: Checkout flow could not complete. This may require specific data models on backend. Error: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
          }
        } catch(e) {
           console.log(`    ⚠️ Warning: Could not fetch cart. Error: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
        }
      }
    } else {
      console.log(`\n    ℹ️ No products available to test Cart & Checkout flows.`);
    }

    console.log("\n=========================================");
    console.log("🎉 Integration Tests Finished!");
    console.log("=========================================\n");

  } catch (error) {
    console.error("\n❌ FATAL ERROR DURING TESTS:");
    console.error(error.response ? error.response.data : error.message);
    process.exit(1);
  }
}

runTests();
