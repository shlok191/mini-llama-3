// index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import WebPage from './pages/vanilla.tsx';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <WebPage />
  </React.StrictMode>
);