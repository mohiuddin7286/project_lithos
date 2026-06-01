import React, { useState } from 'react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Home from './Home.jsx'

function RootComponent() {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <StrictMode>
      {currentPage === 'home' ? (
        <Home onLaunchDashboard={() => setCurrentPage('app')} />
      ) : (
        <App />
      )}
    </StrictMode>
  )
}

createRoot(document.getElementById('root')).render(<RootComponent />)
