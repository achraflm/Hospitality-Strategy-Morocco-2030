import React, { useState, useEffect } from 'react'
import { Check, Settings2, Play, Table, TrendingUp, AlertTriangle, HelpCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

function Forecasting() {
  const [activeTab, setActiveTab] = useState('metrics') // 'metrics' or 'predict'
  const [availableModels, setAvailableModels] = useState([])
  const [availableFeatures, setAvailableFeatures] = useState([])
  
  // Settings state
  const [selectedModels, setSelectedModels] = useState(['SARIMA', 'Ridge', 'LSTM'])
  const [selectedFeatures, setSelectedFeatures] = useState([
    'lags_1', 'lags_2', 'lags_12', 'growth_yoy', 'month_sin', 'month_cos', 
    'year', 'is_high_season', 'cdm_event'
  ])
  const [splitYear, setSplitYear] = useState(2023)
  const [dlEpochs, setDlEpochs] = useState(5)
  
  // Future prediction settings
  const [targetYear, setTargetYear] = useState(2030)
  const [inflationRate, setInflationRate] = useState(2.5) // percent
  const [wcBoost, setWcBoost] = useState(20.0) // percent
  
  // Results state
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState(null)
  const [metricsResult, setMetricsResult] = useState(null) // { metrics: [], chartData: [] }
  
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState(null)
  const [predictResult, setPredictResult] = useState(null) // { projections: [], models: [] }
  const [predictViewType, setPredictViewType] = useState('Arrivals') // 'Arrivals' or 'Receipts'

  // Fetch models and features on mount
  useEffect(() => {
    fetch('/api/forecast/models')
      .then(res => res.json())
      .then(data => setAvailableModels(data))
      .catch(err => console.error('Error fetching models:', err))

    fetch('/api/forecast/features')
      .then(res => res.json())
      .then(data => {
        // If features contain complex lags/roll features, let's sort them nicely
        setAvailableFeatures(data)
      })
      .catch(err => console.error('Error fetching features:', err))
  }, [])

  // Run training & evaluation metrics
  const handleRunMetrics = () => {
    if (selectedModels.length === 0) {
      setMetricsError('Veuillez sélectionner au moins un modèle.')
      return
    }
    if (selectedFeatures.length === 0) {
      setMetricsError('Veuillez sélectionner au moins une caractéristique (feature).')
      return
    }
    
    setMetricsLoading(true)
    setMetricsError(null)
    
    fetch('/api/forecast/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        split_year: Number(splitYear),
        selected_features: selectedFeatures,
        selected_models: selectedModels,
        dl_epochs: Number(dlEpochs)
      })
    })
      .then(async res => {
        if (!res.ok) {
          const detail = await res.json()
          throw new Error(detail.detail || 'Erreur lors du calcul des métriques')
        }
        return res.json()
      })
      .then(data => {
        setMetricsResult(data)
        setMetricsLoading(false)
      })
      .catch(err => {
        console.error(err)
        setMetricsError(err.message)
        setMetricsLoading(false)
      })
  }

  // Run future predictions
  const handleRunPredict = () => {
    if (selectedModels.length === 0) {
      setPredictError('Veuillez sélectionner au moins un modèle.')
      return
    }
    if (selectedFeatures.length === 0) {
      setPredictError('Veuillez sélectionner au moins une caractéristique (feature).')
      return
    }
    
    setPredictLoading(true)
    setPredictError(null)
    
    fetch('/api/forecast/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_year: Number(targetYear),
        selected_features: selectedFeatures,
        selected_models: selectedModels,
        dl_epochs: Number(dlEpochs),
        inflation_rate: Number(inflationRate) / 100,
        wc_boost: Number(wcBoost) / 100
      })
    })
      .then(async res => {
        if (!res.ok) {
          const detail = await res.json()
          throw new Error(detail.detail || 'Erreur lors du calcul des prévisions futures')
        }
        return res.json()
      })
      .then(data => {
        setPredictResult(data)
        setPredictLoading(false)
      })
      .catch(err => {
        console.error(err)
        setPredictError(err.message)
        setPredictLoading(false)
      })
  }

  // Helper colors mapping for lines
  const modelColors = {
    'SARIMA': '#00f2fe',
    'Prophet': '#ffb300',
    'Ridge': '#c77dff',
    'Random Forest': '#f43f5e',
    'Extra Trees': '#10b981',
    'Gradient Boosting': '#00e676',
    'AdaBoost': '#3b82f6',
    'XGBoost': '#ec4899',
    'LightGBM': '#a855f7',
    'CatBoost': '#8b5cf6',
    'SVR': '#14b8a6',
    'LSTM': '#e11d48',
    'SimpleRNN': '#f97316'
  }

  // Toggle list element
  const toggleModel = (model) => {
    if (selectedModels.includes(model)) {
      setSelectedModels(selectedModels.filter(m => m !== model))
    } else {
      setSelectedModels([...selectedModels, model])
    }
  }

  const toggleFeature = (feat) => {
    if (selectedFeatures.includes(feat)) {
      setSelectedFeatures(selectedFeatures.filter(f => f !== feat))
    } else {
      setSelectedFeatures([...selectedFeatures, feat])
    }
  }

  const exportCsv = () => {
    if (!predictResult || !predictResult.projections) return
    const keys = Object.keys(predictResult.projections[0])
    const csvContent = "data:text/csv;charset=utf-8," 
      + [keys.join(",")].concat(
          predictResult.projections.map(row => 
            keys.map(k => typeof row[k] === 'boolean' ? row[k] : String(row[k] || 0)).join(",")
          )
        ).join("\n")
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `Projections_Maroc_${targetYear}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-extrabold text-white tracking-tight">Prévisions Temporelles de la Demande</h2>
        <p className="text-slate-400 mt-1">
          Configurez vos variables économiques et sélectionnez parmi nos 3 meilleurs modèles de Machine Learning & Deep Learning (SARIMA, Ridge, LSTM).
        </p>
      </div>

      {/* Tabs Control */}
      <div className="flex border-b border-[#1f293d] space-x-8">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`pb-4 text-sm font-semibold transition-colors border-b-2 ${
            activeTab === 'metrics'
              ? 'border-accentCyan text-accentCyan'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          1. Évaluation Historique (Test Split)
        </button>
        <button
          onClick={() => setActiveTab('predict')}
          className={`pb-4 text-sm font-semibold transition-colors border-b-2 ${
            activeTab === 'predict'
              ? 'border-accentCyan text-accentCyan'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          2. Projections Futures (2026 - 2035)
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Left Settings Bar */}
        <div className="glass-panel p-5 rounded-2xl space-y-6 lg:col-span-1 h-fit">
          <div className="flex items-center gap-2 border-b border-[#1f293d] pb-3">
            <Settings2 size={16} className="text-accentCyan" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Paramétrage Global</h3>
          </div>

          {/* Model selection */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-400 uppercase">Modèles IA ({selectedModels.length})</label>
            <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
              {availableModels.map(model => (
                <button
                  key={model}
                  onClick={() => toggleModel(model)}
                  className={`flex items-center justify-between w-full text-left px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    selectedModels.includes(model)
                      ? 'bg-accentCyan/10 text-accentCyan border border-accentCyan/20'
                      : 'bg-[#121824] text-slate-400 border border-transparent hover:border-[#1f293d] hover:text-slate-300'
                  }`}
                >
                  <span>{model}</span>
                  {selectedModels.includes(model) && <Check size={12} />}
                </button>
              ))}
            </div>
          </div>

          {/* Feature selection */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-400 uppercase">Variables explicatives ({selectedFeatures.length})</label>
            <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
              {availableFeatures.map(feat => (
                <button
                  key={feat}
                  onClick={() => toggleFeature(feat)}
                  className={`flex items-center justify-between w-full text-left px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    selectedFeatures.includes(feat)
                      ? 'bg-accentGreen/10 text-accentGreen border border-accentGreen/20'
                      : 'bg-[#121824] text-slate-400 border border-transparent hover:border-[#1f293d] hover:text-slate-300'
                  }`}
                >
                  <span className="truncate">{feat}</span>
                  {selectedFeatures.includes(feat) && <Check size={12} />}
                </button>
              ))}
            </div>
          </div>

          {/* Deep Learning Settings */}
          {(selectedModels.includes('LSTM') || selectedModels.includes('SimpleRNN')) && (
            <div className="space-y-2 bg-[#090d16]/50 p-3 rounded-xl border border-[#1f293d]">
              <label className="text-xs font-bold text-accentAmber uppercase">Époques Réseau de Neurones</label>
              <div className="flex items-center justify-between mt-1">
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={dlEpochs}
                  onChange={(e) => setDlEpochs(Number(e.target.value))}
                  className="w-3/4 accent-accentAmber"
                />
                <span className="text-xs text-white font-bold bg-[#1a2233] px-2 py-0.5 rounded-md">
                  {dlEpochs}
                </span>
              </div>
            </div>
          )}

          {activeTab === 'metrics' ? (
            /* Training configurations */
            <div className="space-y-4">
              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-400 uppercase">Année test-split</label>
                <select
                  value={splitYear}
                  onChange={(e) => setSplitYear(Number(e.target.value))}
                  className="w-full bg-[#121824] border border-[#1f293d] rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-accentCyan"
                >
                  <option value={2022}>2022</option>
                  <option value={2023}>2023</option>
                  <option value={2024}>2024</option>
                </select>
              </div>

              <button
                onClick={handleRunMetrics}
                disabled={metricsLoading}
                className="w-full flex items-center justify-center gap-2 bg-accentCyan text-[#090d16] font-bold py-2.5 rounded-xl hover:bg-accentCyan/90 disabled:opacity-50 transition-all duration-200 shadow-[0_0_15px_rgba(0,242,254,0.2)]"
              >
                {metricsLoading ? (
                  <div className="w-5 h-5 border-2 border-t-transparent border-[#090d16] rounded-full animate-spin"></div>
                ) : (
                  <>
                    <Play size={16} fill="currentColor" />
                    <span>Lancer Évaluation</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            /* Prediction configurations */
            <div className="space-y-4">
              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-400 uppercase">Horizon (Année Cible)</label>
                <select
                  value={targetYear}
                  onChange={(e) => setTargetYear(Number(e.target.value))}
                  className="w-full bg-[#121824] border border-[#1f293d] rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-accentCyan"
                >
                  <option value={2028}>2028</option>
                  <option value={2030}>2030</option>
                  <option value={2032}>2032</option>
                  <option value={2035}>2035</option>
                </select>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span className="font-bold uppercase">Inflation Annuelle</span>
                  <span className="text-white font-semibold">{inflationRate}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={inflationRate}
                  onChange={(e) => setInflationRate(Number(e.target.value))}
                  className="w-full accent-accentCyan"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span className="font-bold uppercase">Boost Coupe du Monde 2030</span>
                  <span className="text-white font-semibold">+{wcBoost}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="50"
                  step="0.5"
                  value={wcBoost}
                  onChange={(e) => setWcBoost(Number(e.target.value))}
                  className="w-full accent-accentGreen"
                />
              </div>

              <button
                onClick={handleRunPredict}
                disabled={predictLoading}
                className="w-full flex items-center justify-center gap-2 bg-accentGreen text-[#090d16] font-bold py-2.5 rounded-xl hover:bg-accentGreen/90 disabled:opacity-50 transition-all duration-200 shadow-[0_0_15px_rgba(0,230,118,0.2)]"
              >
                {predictLoading ? (
                  <div className="w-5 h-5 border-2 border-t-transparent border-[#090d16] rounded-full animate-spin"></div>
                ) : (
                  <>
                    <Play size={16} fill="currentColor" />
                    <span>Lancer Projections</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Right Outputs Panel */}
        <div className="lg:col-span-3 space-y-8">
          {activeTab === 'metrics' ? (
            /* METRICS RESULTS WINDOW */
            <div className="space-y-8">
              {metricsError && (
                <div className="flex items-center gap-3 bg-accentRose/10 border border-accentRose/30 text-accentRose p-4 rounded-xl text-sm">
                  <AlertTriangle size={20} className="shrink-0" />
                  <span>{metricsError}</span>
                </div>
              )}

              {!metricsResult && !metricsLoading && (
                <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center space-y-4">
                  <div className="p-4 bg-accentCyan/10 text-accentCyan rounded-full">
                    <Table size={32} />
                  </div>
                  <h3 className="text-lg font-bold text-white">Prêt pour l'Évaluation Historique</h3>
                  <p className="text-slate-400 max-w-md text-sm">
                    Configurez vos modèles et vos variables explicatives puis cliquez sur "Lancer Évaluation" pour comparer les précisions de prédictions sur le test split.
                  </p>
                </div>
              )}

              {metricsLoading && (
                <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center space-y-4">
                  <div className="w-12 h-12 border-4 border-t-accentCyan border-slate-700 rounded-full animate-spin"></div>
                  <h3 className="text-md font-bold text-white">Calcul en cours...</h3>
                  <p className="text-slate-400 text-sm max-w-sm text-center">
                    Entraînement des modèles et évaluation récursive sur la période de test split. Les modèles LSTM/SimpleRNN peuvent nécessiter quelques secondes de calcul.
                  </p>
                </div>
              )}

              {metricsResult && !metricsLoading && (
                <>
                  {/* Chart */}
                  <div className="glass-panel p-6 rounded-2xl space-y-6">
                    <h3 className="text-lg font-bold text-white">Graphique de Comparaison du Split Test</h3>
                    <div className="h-[320px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={metricsResult.chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                          <XAxis dataKey="Date" stroke="#475569" fontSize={11} tickLine={false} />
                          <YAxis stroke="#475569" fontSize={11} tickLine={false} tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`} />
                          <Tooltip contentStyle={{ backgroundColor: '#121824', borderColor: '#1f293d', borderRadius: '12px' }} />
                          <Legend />
                          <Line type="monotone" dataKey="Actual" name="Réel (Actual)" stroke="#ffffff" strokeWidth={2} dot={false} />
                          {selectedModels.map(model => (
                            <Line
                              key={model}
                              type="monotone"
                              dataKey={model}
                              name={model}
                              stroke={modelColors[model] || '#cccccc'}
                              strokeWidth={1.5}
                              dot={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Metrics Table */}
                  <div className="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 className="text-lg font-bold text-white">Métriques de Performance Comparative</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm text-slate-300">
                        <thead className="bg-[#121824] text-xs font-semibold text-slate-400 uppercase border-b border-[#1f293d]">
                          <tr>
                            <th className="px-6 py-3">Modèle</th>
                            <th className="px-6 py-3 text-right">R² (Score)</th>
                            <th className="px-6 py-3 text-right">RMSE (Visiteurs)</th>
                            <th className="px-6 py-3 text-right">MAE (Visiteurs)</th>
                            <th className="px-6 py-3 text-right">MAPE (%)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#1f293d]">
                          {metricsResult.metrics.map((row) => {
                            const isBest = row.Model === metricsResult.metrics.reduce((prev, curr) => prev.RMSE < curr.RMSE ? prev : curr).Model
                            return (
                              <tr key={row.Model} className={`hover:bg-[#1a2233]/40 transition-colors ${isBest ? 'bg-accentGreen/5 border-l-4 border-accentGreen' : ''}`}>
                                <td className="px-6 py-4 font-bold text-white flex items-center gap-2">
                                  <span>🤖</span>
                                  <span>{row.Model}</span>
                                  {isBest && <span className="text-[10px] uppercase font-bold bg-accentGreen/20 text-accentGreen px-1.5 py-0.5 rounded ml-1 border border-accentGreen/30">TOP</span>}
                                </td>
                                <td className={`px-6 py-4 text-right font-semibold ${row.R2 > 0 ? 'text-accentGreen' : 'text-accentRose'}`}>
                                  {row.R2.toFixed(3)}
                                </td>
                                <td className="px-6 py-4 text-right font-semibold">
                                  {row.RMSE.toLocaleString()}
                                </td>
                                <td className="px-6 py-4 text-right">
                                  {row.MAE.toLocaleString()}
                                </td>
                                <td className="px-6 py-4 text-right">
                                  {row.MAPE.toFixed(2)}%
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : (
            /* FUTURE PROJECTIONS WINDOW */
            <div className="space-y-8">
              {predictError && (
                <div className="flex items-center gap-3 bg-accentRose/10 border border-accentRose/30 text-accentRose p-4 rounded-xl text-sm">
                  <AlertTriangle size={20} className="shrink-0" />
                  <span>{predictError}</span>
                </div>
              )}

              {!predictResult && !predictLoading && (
                <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center space-y-4">
                  <div className="p-4 bg-accentGreen/10 text-accentGreen rounded-full">
                    <TrendingUp size={32} />
                  </div>
                  <h3 className="text-lg font-bold text-white">Prêt pour les Projections Futures</h3>
                  <p className="text-slate-400 max-w-md text-sm">
                    Calculez les trajectoires d'arrivées touristiques et de recettes en MDH jusqu'à l'année {targetYear}. Intègre le pic de demande de la Coupe du Monde FIFA 2030.
                  </p>
                </div>
              )}

              {predictLoading && (
                <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center space-y-4">
                  <div className="w-12 h-12 border-4 border-t-accentGreen border-slate-700 rounded-full animate-spin"></div>
                  <h3 className="text-md font-bold text-white">Calcul en cours...</h3>
                  <p className="text-slate-400 text-sm max-w-sm text-center">
                    Génération récursive des prévisions multivariées. Calcul des recettes touristiques théoriques indexées sur l'inflation.
                  </p>
                </div>
              )}

              {predictResult && !predictLoading && (
                <>
                  {/* Chart controls & plot */}
                  <div className="glass-panel p-6 rounded-2xl space-y-6">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div>
                        <h3 className="text-lg font-bold text-white">Trajectoires Multi-Modèles & Scénarios Cible</h3>
                        <p className="text-xs text-slate-400">Arrivées historiques vs prévisions futures (2026 - {targetYear})</p>
                      </div>
                      <div className="flex items-center bg-[#121824] rounded-xl p-1 border border-[#1f293d] shrink-0">
                        <button
                          onClick={() => setPredictViewType('Arrivals')}
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                            predictViewType === 'Arrivals' ? 'bg-accentCyan text-[#090d16]' : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Arrivées (Visiteurs)
                        </button>
                        <button
                          onClick={() => setPredictViewType('Receipts')}
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                            predictViewType === 'Receipts' ? 'bg-accentGreen text-[#090d16]' : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Recettes (MDH)
                        </button>
                      </div>
                    </div>

                    <div className="h-[340px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={predictResult.projections}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                          <XAxis dataKey="Date" stroke="#475569" fontSize={11} tickLine={false} />
                          <YAxis 
                            stroke="#475569" 
                            fontSize={11} 
                            tickLine={false} 
                            tickFormatter={(val) => 
                              predictViewType === 'Arrivals' 
                                ? `${(val / 1e6).toFixed(1)}M` 
                                : `${(val / 1e3).toFixed(0)}k`
                            } 
                          />
                          <Tooltip contentStyle={{ backgroundColor: '#121824', borderColor: '#1f293d', borderRadius: '12px' }} />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey={predictViewType === 'Arrivals' ? 'Actual_Arrivals' : 'Actual_Receipts'}
                            name="Historique Réel"
                            stroke="#ffffff"
                            strokeWidth={2}
                            dot={false}
                          />
                          {predictResult.models.map(model => (
                            <Line
                              key={model}
                              type="monotone"
                              dataKey={`${model}_${predictViewType}`}
                              name={`${model} Projeté`}
                              stroke={modelColors[model] || '#cccccc'}
                              strokeWidth={1.5}
                              strokeDasharray="4 4"
                              dot={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="flex justify-between items-center pt-2">
                      <div className="text-xs text-slate-400 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-accentGreen"></span>
                        Les lignes pointillées représentent les prédictions calculées récursivement.
                      </div>
                      <button
                        onClick={exportCsv}
                        className="bg-[#1a2233] border border-[#1f293d] hover:bg-[#25324c] text-white text-xs font-bold px-4 py-2 rounded-xl transition-all"
                      >
                        Exporter les Données en CSV
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Forecasting
