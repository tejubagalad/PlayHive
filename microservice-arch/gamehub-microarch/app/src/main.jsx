import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// ✅ Load Bootstrap (global responsive grid)
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';

// ✅ Load your styles
import './index.css';
import './App.css';

// ✅ Load main component
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
