// index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import WebPage from './pages/index.tsx';
import { RecordsProvider } from './contexts/records.tsx';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <RecordsProvider> 
      <WebPage />
    </RecordsProvider> 
  </React.StrictMode>
);