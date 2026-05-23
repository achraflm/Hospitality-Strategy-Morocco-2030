import React, { useState } from 'react'
import { LayoutDashboard, TrendingUp, DollarSign, Dices, Menu, X } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import RoiSimulator from './pages/RoiSimulator'
import MonteCarlo from './pages/MonteCarlo'


function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />
      case 'forecasting':
        return <Forecasting />
      case 'roi':
        return <RoiSimulator />
      case 'montecarlo':
        return <MonteCarlo />

      default:
        return <Dashboard />
    }
  }

  const navItems = [
    { id: 'dashboard', name: 'Tableau de Bord', icon: LayoutDashboard },
    { id: 'forecasting', name: 'Prévisions IA', icon: TrendingUp },
    { id: 'roi', name: 'Simulateur ROI', icon: DollarSign },
    { id: 'montecarlo', name: 'Simulation Monte Carlo', icon: Dices },

  ]

  return (
    <div className="flex min-h-screen bg-[#090d16] text-slate-100 font-sans">
      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside 
        className={`fixed inset-y-0 left-0 z-50 flex flex-col w-64 bg-[#121824] border-r border-[#1f293d] transition-transform duration-300 transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="flex items-center justify-between h-20 px-6 border-b border-[#1f293d]">
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">🇲🇦</span>
            <span className="font-bold text-lg tracking-wider bg-gradient-to-r from-accentCyan to-accentGreen bg-clip-text text-transparent">
              MOROCCO 2030
            </span>
          </div>
          <button 
            className="md:hidden text-slate-400 hover:text-white transition-colors" 
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = currentPage === item.id
            return (
              <button
                key={item.id}
                onClick={() => {
                  setCurrentPage(item.id)
                  if (window.innerWidth < 768) setSidebarOpen(false)
                }}
                className={`flex items-center w-full gap-4 px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 ${
                  isActive 
                    ? 'bg-accentCyan/10 text-accentCyan border-l-4 border-accentCyan shadow-[0_0_15px_rgba(0,242,254,0.06)]' 
                    : 'text-slate-400 hover:bg-[#1a2233] hover:text-slate-100'
                }`}
              >
                <Icon size={18} />
                <span>{item.name}</span>
              </button>
            )
          })}
        </nav>

        <div className="p-4 border-t border-[#1f293d] bg-[#090d16]/30">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-accentGreen animate-pulse pulse-glow"></div>
            <div className="text-xs text-slate-400">
              <p className="font-semibold text-slate-300">FIFA World Cup 2030</p>
              <p>Moteur Financier Connecté</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${sidebarOpen ? 'md:pl-64' : ''}`}>
        {/* Top Header */}
        <header className="flex items-center justify-between h-20 px-6 md:px-8 border-b border-[#1f293d] bg-[#121824]/60 backdrop-blur-md sticky top-0 z-40">
          <div className="flex items-center gap-4">
            <button 
              className="text-slate-400 hover:text-white transition-colors" 
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={22} />
            </button>
            <h1 className="text-lg font-bold tracking-tight text-white truncate hidden sm:block">
              Morocco Tourism Investment Intelligence Platform
            </h1>
            <h1 className="text-base font-bold tracking-tight text-white sm:hidden">
              MTI Platform
            </h1>
          </div>
          
          <div className="flex items-center gap-3 shrink-0">
            <div className="hidden lg:flex items-center gap-2 bg-[#1a2233] border border-[#1f293d] rounded-xl px-3 py-1.5 text-xs text-slate-400">
              <span className="font-semibold text-accentGreen">API Backend:</span>
              <span>Online</span>
            </div>
            <div className="flex items-center gap-2 bg-[#1a2233] border border-[#1f293d] rounded-xl px-3 py-1.5 text-xs font-semibold text-accentCyan">
              <span>FIFA 2030 Engine</span>
            </div>
          </div>
        </header>

        {/* Content Wrapper */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto">
          {renderPage()}
        </main>
      </div>
    </div>
  )
}

export default App
