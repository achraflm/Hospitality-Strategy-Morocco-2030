import React, { useState, useEffect } from 'react'
import { DollarSign, Sliders, CheckCircle, TrendingUp, AlertCircle, FileText, ChevronRight } from 'lucide-react'

function RoiSimulator() {
  const [cities, setCities] = useState({})
  const [selectedCity, setSelectedCity] = useState('Marrakech')
  
  // Sliders state
  const [rooms, setRooms] = useState(200)
  const [capexM, setCapexM] = useState(150)
  const [baseAdr, setBaseAdr] = useState(250)
  const [baseOccupancy, setBaseOccupancy] = useState(65) // percent
  const [discountRate, setDiscountRate] = useState(8) // percent
  const [opexMargin, setOpexMargin] = useState(65) // percent
  const [wcAdrBoost, setWcAdrBoost] = useState(40) // percent
  const [inflationRate, setInflationRate] = useState(2.5) // percent
  const [wcOpexInflation, setWcOpexInflation] = useState(6.3) // percent
  const [enableWc, setEnableWc] = useState(true)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null) // { metrics: {}, cashFlows: [] }

  // Load cities on mount
  useEffect(() => {
    fetch('/api/roi/cities')
      .then(res => res.json())
      .then(data => {
        setCities(data)
        // Set default values based on Marrakech baseline
        if (data.Marrakech) {
          setCapexM(data.Marrakech.capex)
          setBaseAdr(data.Marrakech.adr)
        }
      })
      .catch(err => console.error('Error fetching cities:', err))
  }, [])

  // Update sliders based on selected city baseline
  const handleCityChange = (cityName) => {
    setSelectedCity(cityName)
    const cityData = cities[cityName]
    if (cityData) {
      setCapexM(cityData.capex)
      setBaseAdr(cityData.adr)
    }
  }

  // Trigger ROI simulation on parameter change
  useEffect(() => {
    setLoading(true)
    setError(null)

    const timer = setTimeout(() => {
      fetch('/api/roi/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          city: selectedCity,
          rooms: Number(rooms),
          capex_m: Number(capexM),
          base_adr: Number(baseAdr),
          base_occupancy: Number(baseOccupancy) / 100,
          discount_rate: Number(discountRate) / 100,
          opex_margin: Number(opexMargin) / 100,
          wc_adr_boost: Number(wcAdrBoost) / 100,
          wc_occ_boost: 0.15, // Constante dans l'API par défaut
          inflation_rate: Number(inflationRate) / 100,
          wc_opex_inflation: Number(wcOpexInflation) / 100,
          enable_wc: enableWc
        })
      })
        .then(async res => {
          if (!res.ok) {
            const detail = await res.json()
            throw new Error(detail.detail || 'Erreur lors de la simulation ROI')
          }
          return res.json()
        })
        .then(data => {
          setResult(data)
          setLoading(false)
        })
        .catch(err => {
          console.error(err)
          setError(err.message)
          setLoading(false)
        })
    }, 300) // Debounce API calls

    return () => clearTimeout(timer)
  }, [selectedCity, rooms, capexM, baseAdr, baseOccupancy, discountRate, opexMargin, wcAdrBoost, inflationRate, wcOpexInflation, enableWc])

  const exportCashFlows = () => {
    if (!result || !result.cashFlows) return
    const keys = Object.keys(result.cashFlows[0])
    const csvContent = "data:text/csv;charset=utf-8," 
      + [keys.join(",")].concat(
          result.cashFlows.map(row => keys.map(k => String(row[k])).join(","))
        ).join("\n")
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `Feuille_Cash_Flow_${selectedCity}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-extrabold text-white tracking-tight">Simulateur ROI Hôtelier Stratégique</h2>
        <p className="text-slate-400 mt-1">
          Calculez la faisabilité économique, la VAN, le TRI et les cash flows sur 10 ans avec prise en compte du choc Coupe du Monde 2030.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-accentRose/10 border border-accentRose/30 text-accentRose p-4 rounded-xl text-sm">
          <AlertCircle size={20} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sliders Control Panel */}
        <div className="glass-panel p-5 rounded-2xl space-y-6 lg:col-span-1 h-fit">
          <div className="flex items-center gap-2 border-b border-[#1f293d] pb-3">
            <Sliders size={16} className="text-accentCyan" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Paramètres de l'Hôtel</h3>
          </div>

          {/* City Selection */}
          <div className="space-y-1">
            <label className="text-xs font-bold text-slate-400 uppercase">Ville de Destination</label>
            <select
              value={selectedCity}
              onChange={(e) => handleCityChange(e.target.value)}
              className="w-full bg-[#121824] border border-[#1f293d] rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-accentCyan"
            >
              {Object.keys(cities).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          {/* Rooms */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">Nombre de Chambres</span>
              <span className="text-white font-semibold">{rooms}</span>
            </div>
            <input
              type="range"
              min="50"
              max="500"
              step="10"
              value={rooms}
              onChange={(e) => setRooms(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* CaPex */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">CaPex (Investissement)</span>
              <span className="text-white font-semibold">{capexM} M$</span>
            </div>
            <input
              type="range"
              min="20"
              max="350"
              step="5"
              value={capexM}
              onChange={(e) => setCapexM(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* ADR */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">ADR Référence ($)</span>
              <span className="text-white font-semibold">{baseAdr} $</span>
            </div>
            <input
              type="range"
              min="50"
              max="800"
              step="10"
              value={baseAdr}
              onChange={(e) => setBaseAdr(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* Occupancy */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">Taux d'Occupation de Base</span>
              <span className="text-white font-semibold">{baseOccupancy}%</span>
            </div>
            <input
              type="range"
              min="30"
              max="95"
              step="1"
              value={baseOccupancy}
              onChange={(e) => setBaseOccupancy(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* WACC Discount Rate */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">Taux d'Actualisation (WACC)</span>
              <span className="text-white font-semibold">{discountRate}%</span>
            </div>
            <input
              type="range"
              min="4"
              max="16"
              step="0.5"
              value={discountRate}
              onChange={(e) => setDiscountRate(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* OpEx Margin */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span className="font-bold uppercase">Marge Opérationnelle (OpEx)</span>
              <span className="text-white font-semibold">{opexMargin}%</span>
            </div>
            <input
              type="range"
              min="35"
              max="80"
              step="1"
              value={opexMargin}
              onChange={(e) => setOpexMargin(Number(e.target.value))}
              className="w-full accent-accentCyan"
            />
          </div>

          {/* World Cup Boost Settings */}
          <div className="space-y-4 pt-3 border-t border-[#1f293d]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase">Activer Coupe du Monde 2030</span>
              <input
                type="checkbox"
                checked={enableWc}
                onChange={(e) => setEnableWc(e.target.checked)}
                className="w-4 h-4 rounded bg-[#121824] border-[#1f293d] accent-accentGreen"
              />
            </div>

            {enableWc && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span className="font-bold text-accentGreen uppercase">Boost ADR Coupe du Monde</span>
                  <span className="text-accentGreen font-semibold">+{wcAdrBoost}%</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="80"
                  step="5"
                  value={wcAdrBoost}
                  onChange={(e) => setWcAdrBoost(Number(e.target.value))}
                  className="w-full accent-accentGreen"
                />
              </div>
            )}

            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span className="font-bold uppercase">Taux d'Inflation Annuel</span>
                <span className="text-white font-semibold">{inflationRate}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="8.0"
                step="0.1"
                value={inflationRate}
                onChange={(e) => setInflationRate(Number(e.target.value))}
                className="w-full accent-accentCyan"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span className="font-bold uppercase">Choc Inflation OPEX (2030)</span>
                <span className="text-white font-semibold">+{wcOpexInflation}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="20"
                step="0.1"
                value={wcOpexInflation}
                onChange={(e) => setWcOpexInflation(Number(e.target.value))}
                className="w-full accent-accentRose"
              />
              <div className="text-[10px] text-slate-500 mt-1">
                Recommandé: 6.3% (Moyenne historique 2014, 2018, 2022)
              </div>
            </div>
          </div>
        </div>

        {/* Outputs: Metrics Comparisons & Cash Flow Sheet */}
        <div className="lg:col-span-2 space-y-8">
          {/* Comparison Cards */}
          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Base Scenario Card */}
              <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border-t-4 border-slate-500">
                <h3 className="text-slate-400 text-sm font-bold uppercase">Scénario de Référence (Base)</h3>
                
                <div className="mt-4 space-y-3">
                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">Valeur Actuelle Net (VAN) :</span>
                    <span className={`text-xl font-bold ${result.metrics.NPV_Base >= 0 ? 'text-accentGreen' : 'text-accentRose'}`}>
                      {result.metrics.NPV_Base.toFixed(2)} M$
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">Taux de Rentabilité Interne (TRI) :</span>
                    <span className="text-lg font-bold text-slate-200">
                      {result.metrics.IRR_Base !== null ? `${result.metrics.IRR_Base.toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>

                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">Délai de Récupération (Payback) :</span>
                    <span className="text-sm font-semibold text-slate-300">
                      {result.metrics.Payback_Base !== null ? `${result.metrics.Payback_Base} ans` : 'Non rentabilisé'}
                    </span>
                  </div>

                  <div className="flex justify-between items-baseline border-t border-[#1f293d] pt-2 mt-2">
                    <span className="text-xs text-slate-400">ROI Cumulé :</span>
                    <span className="text-lg font-bold text-slate-200">
                      {result.metrics.ROI_Base.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* World Cup Scenario Card */}
              <div className="glass-panel p-6 rounded-2xl relative overflow-hidden border-t-4 border-accentGreen shadow-[0_0_20px_rgba(0,230,118,0.04)]">
                <div className="absolute top-0 right-0 w-16 h-16 bg-accentGreen/5 rounded-bl-3xl flex items-center justify-center">
                  <span className="text-xs">🏆</span>
                </div>
                <h3 className="text-accentGreen text-sm font-bold uppercase flex items-center gap-1.5">
                  Scénario Coupe du Monde 2030
                </h3>
                
                <div className="mt-4 space-y-3">
                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">VAN Estimée :</span>
                    <span className={`text-xl font-bold ${result.metrics.NPV_WC >= 0 ? 'text-accentGreen' : 'text-accentRose'}`}>
                      {result.metrics.NPV_WC.toFixed(2)} M$
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">TRI Stimulé :</span>
                    <span className="text-lg font-bold text-accentGreen">
                      {result.metrics.IRR_WC !== null ? `${result.metrics.IRR_WC.toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>

                  <div className="flex justify-between items-baseline">
                    <span className="text-xs text-slate-400">Délai de Récupération :</span>
                    <span className="text-sm font-semibold text-accentGreen">
                      {result.metrics.Payback_WC !== null ? `${result.metrics.Payback_WC} ans` : 'Non rentabilisé'}
                    </span>
                  </div>

                  <div className="flex justify-between items-baseline border-t border-[#1f293d] pt-2 mt-2">
                    <span className="text-xs text-slate-400">ROI Cumulé Stimulé :</span>
                    <span className="text-lg font-bold text-accentGreen">
                      {result.metrics.ROI_WC.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Cash Flow Table */}
          {result && (
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-bold text-white">Feuille de Cash Flow sur 10 ans</h3>
                  <p className="text-xs text-slate-400 mt-0.5">Valeurs de revenus et GOP exprimées en Millions USD (M$)</p>
                </div>
                <button
                  onClick={exportCashFlows}
                  className="bg-[#1a2233] border border-[#1f293d] hover:bg-[#25324c] text-white text-xs font-bold px-3 py-2 rounded-xl transition-all"
                >
                  Exporter CSV
                </button>
              </div>

              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-[#121824] text-[#94a3b8] font-bold uppercase sticky top-0 border-b border-[#1f293d]">
                    <tr>
                      <th className="px-4 py-3">Année</th>
                      <th className="px-4 py-3 text-right">ADR Base ($)</th>
                      <th className="px-4 py-3 text-right">Occ Base (%)</th>
                      <th className="px-4 py-3 text-right">Revenu Base (M$)</th>
                      <th className="px-4 py-3 text-right text-accentGreen">ADR CDM ($)</th>
                      <th className="px-4 py-3 text-right text-accentGreen">Occ CDM (%)</th>
                      <th className="px-4 py-3 text-right text-accentGreen">Revenu CDM (M$)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f293d]">
                    {result.cashFlows.map((row) => {
                      const isWorldCupYear = row.Year === 2030
                      return (
                        <tr key={row.Year} className={`hover:bg-[#1a2233]/40 transition-colors ${isWorldCupYear ? 'bg-accentGreen/5 border-y border-accentGreen/30' : ''}`}>
                          <td className="px-4 py-3 font-bold text-white flex items-center gap-1">
                            <span>{row.Year}</span>
                            {isWorldCupYear && <span className="text-[9px] bg-accentGreen/25 text-accentGreen px-1 rounded font-normal">WC</span>}
                          </td>
                          <td className="px-4 py-3 text-right">{row.ADR_Base.toFixed(0)} $</td>
                          <td className="px-4 py-3 text-right">{row.Occ_Base.toFixed(1)}%</td>
                          <td className="px-4 py-3 text-right font-semibold text-slate-200">{row.Revenue_Base.toFixed(2)} M$</td>
                          <td className="px-4 py-3 text-right text-accentGreen font-medium">{row.ADR_WC.toFixed(0)} $</td>
                          <td className="px-4 py-3 text-right text-accentGreen font-medium">{row.Occ_WC.toFixed(1)}%</td>
                          <td className="px-4 py-3 text-right text-accentGreen font-bold">{row.Revenue_WC.toFixed(2)} M$</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RoiSimulator
