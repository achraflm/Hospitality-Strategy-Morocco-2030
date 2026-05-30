import React, { useState, useEffect } from 'react'
import { TrendingUp, Award, MapPin, DollarSign, Building, Percent, RefreshCw, AlertCircle } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

function Dashboard() {
  const [cities, setCities] = useState({})
  const [loadingCities, setLoadingCities] = useState(true)
  const [forecastData, setForecastData] = useState([])
  const [loadingForecast, setLoadingForecast] = useState(true)
  const [error, setError] = useState(null)
  
  useEffect(() => {
    // Fetch cities baseline data
    fetch('/api/roi/cities')
      .then(res => {
        if (!res.ok) throw new Error('Erreur de chargement des données de villes')
        return res.json()
      })
      .then(data => {
        setCities(data)
        setLoadingCities(false)
      })
      .catch(err => {
        console.error(err)
        setError(err.message)
        setLoadingCities(false)
      })

    // Fetch default projection to show on the dashboard (CatBoost model, year 2030)
    const defaultFeatures = [
      'lags_1', 'lags_2', 'lags_12', 'growth_yoy', 'month_sin', 'month_cos', 
      'year', 'is_high_season', 'cdm_event'
    ]
    
    fetch('/api/forecast/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_year: 2030,
        selected_features: defaultFeatures,
        selected_models: ['XGBoost', 'LSTM', 'GRU'],
        dl_epochs: 5,
        inflation_rate: 0.015,
        wc_boost: 0.20
      })
    })
      .then(res => {
        if (!res.ok) throw new Error('Erreur de chargement des prévisions')
        return res.json()
      })
      .then(data => {
        // Recharts data format: { Date, Actual_Arrivals, XGBoost_Arrivals, LSTM_Arrivals, GRU_Arrivals }
        setForecastData(data.projections || [])
        setLoadingForecast(false)
      })
      .catch(err => {
        console.error(err)
        setError(err.message)
        setLoadingForecast(false)
      })
  }, [])

  // Calculate sum of projected arrivals in 2030
  const get2030ArrivalsProjection = () => {
    if (!forecastData.length) return '17.5M'
    // Filter only future projections in year 2030
    const arrivals2030 = forecastData.filter(d => d.IsFuture && d.Date.startsWith('2030'))
    if (!arrivals2030.length) return '17.5M'
    
    // Sum for the model XGBoost
    const sum = arrivals2030.reduce((acc, curr) => acc + (curr.XGBoost_Arrivals || 0), 0)
    if (sum === 0) return '17.5M'
    return `${(sum / 1e6).toFixed(2)}M`
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 animate-fade-in-up">
        <div>
          <h2 className="text-2xl font-extrabold text-white tracking-tight">Vue d'ensemble analytique</h2>
          <p className="text-slate-400 mt-1">Données touristiques marocaines, projections d'investissements et préparation à la Coupe du Monde FIFA 2030.</p>
        </div>
        <button 
          onClick={() => window.location.reload()} 
          className="flex items-center gap-2 bg-[#1a2233] border border-[#1f293d] hover:bg-[#25324c] text-slate-300 hover:text-white px-4 py-2 rounded-xl text-sm transition-all duration-200"
        >
          <RefreshCw size={16} />
          <span>Actualiser</span>
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-accentRose/10 border border-accentRose/30 text-accentRose p-4 rounded-xl text-sm">
          <AlertCircle size={20} className="shrink-0" />
          <span>{error} - Assurez-vous que le serveur API est lancé en arrière plan (`python backend/main.py`)</span>
        </div>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* KPI 1 */}
        <div className="glass-panel glass-panel-hover p-6 rounded-2xl relative overflow-hidden group opacity-0 animate-fade-in-up delay-100">
          <div className="absolute top-0 right-0 w-24 h-24 bg-accentCyan/5 rounded-full blur-2xl group-hover:bg-accentCyan/10 transition-all duration-500"></div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-400">Objectif Arrivées 2030</span>
            <div className="p-2.5 bg-accentCyan/10 text-accentCyan rounded-xl">
              <TrendingUp size={20} />
            </div>
          </div>
          <div className="mt-4">
            <h3 className="text-3xl font-extrabold text-white tracking-tight">
              {loadingForecast ? <div className="h-9 w-28 bg-[#1f293d] animate-pulse rounded-md"></div> : get2030ArrivalsProjection()}
            </h3>
            <p className="text-xs text-accentGreen font-medium mt-2 flex items-center gap-1">
              <span>+32% par rapport à 2024</span>
            </p>
          </div>
        </div>

        {/* KPI 2 */}
        <div className="glass-panel glass-panel-hover p-6 rounded-2xl relative overflow-hidden group opacity-0 animate-fade-in-up delay-200">
          <div className="absolute top-0 right-0 w-24 h-24 bg-accentGreen/5 rounded-full blur-2xl group-hover:bg-accentGreen/10 transition-all duration-500"></div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-400">Ville Prioritaire</span>
            <div className="p-2.5 bg-accentGreen/10 text-accentGreen rounded-xl">
              <MapPin size={20} />
            </div>
          </div>
          <div className="mt-4">
            <h3 className="text-3xl font-extrabold text-white tracking-tight">Marrakech</h3>
            <p className="text-xs text-slate-400 font-medium mt-2">
              Demande hôtelière critique (35% part de marché)
            </p>
          </div>
        </div>

        {/* KPI 3 */}
        <div className="glass-panel glass-panel-hover p-6 rounded-2xl relative overflow-hidden group opacity-0 animate-fade-in-up delay-300">
          <div className="absolute top-0 right-0 w-24 h-24 bg-accentAmber/5 rounded-full blur-2xl group-hover:bg-accentAmber/10 transition-all duration-500"></div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-400">Coupe du Monde Boost</span>
            <div className="p-2.5 bg-accentAmber/10 text-accentAmber rounded-xl">
              <Award size={20} />
            </div>
          </div>
          <div className="mt-4">
            <h3 className="text-3xl font-extrabold text-white tracking-tight">+40%</h3>
            <p className="text-xs text-slate-400 font-medium mt-2">
              ADR estimée en 2030 dans les grandes villes
            </p>
          </div>
        </div>

        {/* KPI 4 */}
        <div className="glass-panel glass-panel-hover p-6 rounded-2xl relative overflow-hidden group opacity-0 animate-fade-in-up delay-400">
          <div className="absolute top-0 right-0 w-24 h-24 bg-accentRose/5 rounded-full blur-2xl group-hover:bg-accentRose/10 transition-all duration-500"></div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-400">WACC Moyen National</span>
            <div className="p-2.5 bg-accentRose/10 text-accentRose rounded-xl">
              <Percent size={20} />
            </div>
          </div>
          <div className="mt-4">
            <h3 className="text-3xl font-extrabold text-white tracking-tight">8.0%</h3>
            <p className="text-xs text-slate-400 font-medium mt-2">
              Taux d'actualisation de référence hôtelier
            </p>
          </div>
        </div>
      </div>

      {/* Main Charts & Highlights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Forecast Chart Panel */}
        <div className="glass-panel p-6 rounded-2xl lg:col-span-2 space-y-6 opacity-0 animate-fade-in-up delay-200">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div>
              <h3 className="text-lg font-bold text-white">Trajectoire des Arrivées Touristiques (Mensuel)</h3>
              <p className="text-xs text-slate-400">Arrivées historiques réelles et prévisionnelles récursives (modèle XGBoost)</p>
            </div>
            <div className="flex items-center gap-4 text-xs font-semibold">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-accentCyan"></div>
                <span className="text-slate-300">Historique Réel</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-accentGreen"></div>
                <span className="text-slate-300">Prévision 2026-2030</span>
              </div>
            </div>
          </div>

          <div className="h-[340px] w-full">
            {loadingForecast ? (
              <div className="w-full h-full flex flex-col items-center justify-center gap-3">
                <div className="w-10 h-10 border-4 border-t-accentCyan border-slate-700 rounded-full animate-spin"></div>
                <span className="text-sm text-slate-400 font-medium">Chargement des données de prévision...</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={forecastData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f2fe" stopOpacity={0.25}/>
                      <stop offset="95%" stopColor="#00f2fe" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00e676" stopOpacity={0.25}/>
                      <stop offset="95%" stopColor="#00e676" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                  <XAxis 
                    dataKey="Date" 
                    stroke="#475569" 
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(str) => {
                      if (!str) return ''
                      const parts = str.split('-')
                      return parts[1] === '01' ? parts[0] : ''
                    }}
                  />
                  <YAxis 
                    stroke="#475569" 
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(val) => `${(val / 1e6).toFixed(1)}M`} 
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#121824', borderColor: '#1f293d', borderRadius: '12px' }}
                    labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                    formatter={(value, name) => {
                      const formatted = (value / 1e3).toFixed(0) + 'k'
                      if (name === 'Actual_Arrivals') return [formatted, 'Arrivées Réelles']
                      if (name === 'XGBoost_Arrivals') return [formatted, 'Prévisions XGBoost']
                      if (name === 'LSTM_Arrivals') return [formatted, 'Prévisions LSTM']
                      if (name === 'GRU_Arrivals') return [formatted, 'Prévisions GRU']
                      return [formatted, name]
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="Actual_Arrivals" 
                    stroke="#00f2fe" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorActual)" 
                    dot={false}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="XGBoost_Arrivals" 
                    stroke="#00e676" 
                    strokeWidth={2} 
                    strokeDasharray="4 4"
                    fillOpacity={1} 
                    fill="url(#colorForecast)" 
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Highlights Side Panel */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white">Impact Macro-économique 2030</h3>
            <p className="text-xs text-slate-400 mt-0.5">Retombées prévues de la Coupe du Monde FIFA 2030</p>
          </div>

          <div className="space-y-5 flex-1 mt-4">
            <div className="flex gap-4">
              <div className="p-2.5 h-10 w-10 bg-accentCyan/10 text-accentCyan rounded-xl flex items-center justify-center shrink-0">
                <Building size={18} />
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-200">Expansion Hôtelière</h4>
                <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                  Planification gouvernementale visant à ajouter 150 000 lits dans les principales villes hôtes de la Coupe du Monde.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="p-2.5 h-10 w-10 bg-accentGreen/10 text-accentGreen rounded-xl flex items-center justify-center shrink-0">
                <DollarSign size={18} />
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-200">Afflux Touristique</h4>
                <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                  Augmentation ponctuelle projetée de +2.5 millions de visiteurs directs durant la période du tournoi.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="p-2.5 h-10 w-10 bg-accentAmber/10 text-accentAmber rounded-xl flex items-center justify-center shrink-0">
                <Percent size={18} />
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-200">Booster de Rentabilité</h4>
                <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                  Taux moyen journalier (ADR) boosté jusqu'à +40% lors des pics saisonniers durant l'année de la Coupe du Monde.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a2233] border border-[#1f293d] rounded-xl p-4 text-xs text-slate-300">
            <p className="font-semibold text-white flex items-center gap-1.5 mb-1">
              <span className="w-1.5 h-1.5 bg-accentAmber rounded-full animate-ping"></span>
              Note de l'architecte :
            </p>
            Les analyses intègrent des scénarios probabilistes combinant l'inflation touristique et les chocs de demande pour calibrer les risques.
          </div>
        </div>
      </div>

      {/* City Investment Table Panel */}
      <div className="glass-panel p-6 rounded-2xl space-y-6 opacity-0 animate-fade-in-up delay-400">
        <div>
          <h3 className="text-lg font-bold text-white">Référentiel & Baselines par Ville</h3>
          <p className="text-xs text-slate-400 mt-0.5">Hypothèses par défaut pour les investissements hôteliers 4 et 5 étoiles au Maroc</p>
        </div>

        <div className="overflow-x-auto">
          {loadingCities ? (
            <div className="space-y-3 py-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-10 bg-[#121824] animate-pulse rounded-md"></div>
              ))}
            </div>
          ) : (
            <table className="w-full text-left text-sm text-slate-300">
              <thead className="bg-[#121824]/80 text-xs font-semibold text-slate-400 uppercase border-b border-[#1f293d]">
                <tr>
                  <th className="px-6 py-4">Ville</th>
                  <th className="px-6 py-4 text-right">CaPex Moyen (M$)</th>
                  <th className="px-6 py-4 text-right">ADR Référence ($)</th>
                  <th className="px-6 py-4 text-right">Part Touristique (%)</th>
                  <th className="px-6 py-4">Recommandation Stratégique</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f293d]">
                {Object.keys(cities).map((name) => {
                  const city = cities[name]
                  return (
                    <tr key={name} className="hover:bg-[#1a2233]/40 transition-colors">
                      <td className="px-6 py-4 font-bold text-white flex items-center gap-2">
                        <span>🇲🇦</span>
                        <span>{name}</span>
                      </td>
                      <td className="px-6 py-4 text-right font-semibold text-slate-200">
                        {city.capex.toFixed(1)} M$
                      </td>
                      <td className="px-6 py-4 text-right font-semibold text-slate-200">
                        {city.adr.toFixed(1)} $
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="bg-[#121824] px-2.5 py-1 rounded-lg text-xs font-semibold text-accentCyan border border-[#1f293d]">
                          {(city.part * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                          city.rec.includes('Prioritaire') 
                            ? 'bg-accentGreen/10 text-accentGreen border border-accentGreen/20'
                            : city.rec.includes('affaires') || city.rec.includes('Saisonnier')
                            ? 'bg-accentCyan/10 text-accentCyan border border-accentCyan/20'
                            : city.rec.includes('Attendre')
                            ? 'bg-accentAmber/10 text-accentAmber border border-accentAmber/20'
                            : 'bg-accentRose/10 text-accentRose border border-accentRose/20'
                        }`}>
                          {city.rec}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
