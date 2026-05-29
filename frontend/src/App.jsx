import React, { useState } from 'react'
import { LayoutDashboard, TrendingUp, BarChart3, LineChart, Target, CalendarClock, FileText, Menu, X, Download, Filter } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import RoiSimulator from './pages/RoiSimulator'
import MonteCarlo from './pages/MonteCarlo'

function App() {
  const [currentPage, setCurrentPage] = useState('models')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const navItems = [
    { id: 'overview', name: 'Overview', icon: LayoutDashboard },
    { id: 'models', name: 'Forecasting Models', icon: TrendingUp },
    { id: 'arrivals', name: 'Arrivals Analysis', icon: BarChart3 },
    { id: 'nights', name: 'Nights Analysis', icon: LineChart },
    { id: 'metrics', name: 'Metrics Comparison', icon: Target },
    { id: 'walkforward', name: 'Walk-forward Validation', icon: CalendarClock },
    { id: 'roi', name: 'ROI Calculator', icon: FileText },
    { id: 'montecarlo', name: 'Monte Carlo', icon: FileText },
  ]

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <Dashboard />
      case 'models':
      case 'arrivals':
      case 'nights':
      case 'metrics':
      case 'walkforward':
        return <Forecasting activeSection={currentPage} />
      case 'roi':
        return <RoiSimulator />
      case 'montecarlo':
        return <MonteCarlo />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="flex min-h-screen bg-[#020617] text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* Sidebar overlay for mobile */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside 
        className={`fixed inset-y-0 left-0 z-50 flex flex-col w-64 bg-[#0f172a]/95 backdrop-blur-xl border-r border-slate-800 transition-transform duration-300 ease-in-out transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="flex items-center justify-between h-20 px-6 border-b border-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <span className="text-sm font-bold text-white">AI</span>
            </div>
            <span className="font-bold text-lg tracking-wide bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              NEXUS
            </span>
          </div>
          <button 
            className="md:hidden text-slate-400 hover:text-white transition-colors" 
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-3 py-6 space-y-1 overflow-y-auto custom-scrollbar">
          <div className="px-3 mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">Analytics</div>
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
                className={`group flex items-center w-full gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${
                  isActive 
                    ? 'bg-indigo-500/10 text-indigo-400' 
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                }`}
              >
                <Icon size={18} className={isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300 transition-colors'} />
                <span>{item.name}</span>
                {isActive && (
                  <motion.div 
                    layoutId="sidebar-active-indicator"
                    className="absolute left-0 w-1 h-8 bg-indigo-500 rounded-r-full"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.2 }}
                  />
                )}
              </button>
            )
          })}
        </nav>

        <div className="p-4 border-t border-slate-800/50">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 border border-slate-800">
            <div className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </div>
            <div className="text-xs">
              <p className="font-medium text-slate-200">System Status</p>
              <p className="text-slate-500">All models online</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${sidebarOpen ? 'md:pl-64' : ''}`}>
        {/* Top Header */}
        <header className="flex items-center justify-between h-20 px-6 lg:px-8 border-b border-slate-800/50 bg-[#020617]/80 backdrop-blur-xl sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button 
              className="text-slate-400 hover:text-white transition-colors md:hidden" 
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={20} />
            </button>
            <h1 className="text-xl font-semibold tracking-tight text-white hidden sm:block">
              {navItems.find(i => i.id === currentPage)?.name || 'Dashboard'}
            </h1>
          </div>
          
          <div className="flex items-center gap-4 shrink-0">
            {/* Top Navbar Actions */}
            <div className="hidden lg:flex items-center gap-2">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/50 border border-slate-700/50 text-sm text-slate-300 cursor-pointer hover:bg-slate-800 transition-colors">
                <Filter size={14} className="text-slate-400" />
                <span>Tourism_Morocco_2026.csv</span>
              </div>
            </div>
            <div className="h-6 w-px bg-slate-800 hidden lg:block"></div>
            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-all shadow-lg shadow-indigo-500/20 active:scale-95">
              <Download size={16} />
              <span className="hidden sm:inline">Export Report</span>
            </button>
          </div>
        </header>

        {/* Content Wrapper */}
        <main className="flex-1 p-6 lg:p-8 overflow-x-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentPage}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {renderPage()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}

export default App
