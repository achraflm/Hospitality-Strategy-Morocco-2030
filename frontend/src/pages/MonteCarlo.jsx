import React, { useState } from 'react'
import { Dices, ShieldAlert, BarChart3, AlertCircle, Percent, TrendingUp, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

function MonteCarlo() {
  // Inputs state
  const [simCount, setSimCount] = useState(500)
  const [rooms, setRooms] = useState(200)
  const [capexM, setCapexM] = useState(150)
  const [baseAdr, setBaseAdr] = useState(250)
  const [baseOccupancy, setBaseOccupancy] = useState(65)
  const [discountRate, setDiscountRate] = useState(8)
  const [opexMargin, setOpexMargin] = useState(65)
  const [wcAdrBoost, setWcAdrBoost] = useState(40)
  const [inflationRate, setInflationRate] = useState(2.5)
  const [enableWc, setEnableWc] = useState(true)

  // Outputs state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null) // { summary: {}, histogram: [] }

  const handleRunSimulation = () => {
    setLoading(true)
    setError(null)
    
    fetch('/api/monte-carlo/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rooms: Number(rooms),
        capex_m: Number(capexM),
        base_adr: Number(baseAdr),
        base_occupancy: Number(baseOccupancy) / 100,
        discount_rate: Number(discountRate) / 100,
        opex_margin: Number(opexMargin) / 100,
        wc_adr_boost: Number(wcAdrBoost) / 100,
        inflation_rate: Number(inflationRate) / 100,
        simulations_count: Number(simCount),
        enable_wc: enableWc
      })
    })
      .then(async res => {
        if (!res.ok) {
          const detail = await res.json()
          throw new Error(detail.detail || 'Erreur lors du calcul Monte Carlo')
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
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-extrabold text-white tracking-tight">Analyse de Risque par Simulation Monte Carlo</h2>
        <p className="text-slate-400 mt-1">
          Modélisez l'incertitude sur l'inflation, l'occupation et les boosts de la Coupe du Monde via des tirages aléatoires répétés.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-accentRose/10 border border-accentRose/30 text-accentRose p-4 rounded-xl text-sm">
          <AlertCircle size={20} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Left Side: Parameters Form */}
        <div className="glass-panel p-5 rounded-2xl space-y-5 lg:col-span-1 h-fit">
          <div className="flex items-center gap-2 border-b border-[#1f293d] pb-3">
            <Dices size={18} className="text-accentAmber animate-pulse" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Moteur Stochastique</h3>
          </div>

          {/* Number of iterations */}
          <div className="space-y-1">
            <label className="text-xs font-bold text-slate-400 uppercase">Nombre d'itérations</label>
            <select
              value={simCount}
              onChange={(e) => setSimCount(Number(e.target.value))}
              className="w-full bg-[#121824] border border-[#1f293d] rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-accentCyan"
            >
              <option value={200}>200 tirages (Rapide)</option>
              <option value={500}>500 tirages (Recommandé)</option>
              <option value={1000}>1000 tirages (Précis)</option>
            </select>
          </div>

          <div className="space-y-4 pt-2 border-t border-[#1f293d]">
            {/* Rooms */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span className="font-bold uppercase">Chambres</span>
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
                <span className="font-bold uppercase">CaPex</span>
                <span className="text-white font-semibold">{capexM} M$</span>
              </div>
              <input
                type="range"
                min="20"
                max="300"
                step="5"
                value={capexM}
                onChange={(e) => setCapexM(Number(e.target.value))}
                className="w-full accent-accentCyan"
              />
            </div>

            {/* ADR */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span className="font-bold uppercase">ADR ($)</span>
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
                <span className="font-bold uppercase">Occupation de base</span>
                <span className="text-white font-semibold">{baseOccupancy}%</span>
              </div>
              <input
                type="range"
                min="30"
                max="90"
                value={baseOccupancy}
                onChange={(e) => setBaseOccupancy(Number(e.target.value))}
                className="w-full accent-accentCyan"
              />
            </div>

            {/* OpEx Margin */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span className="font-bold uppercase">Marge OpEx</span>
                <span className="text-white font-semibold">{opexMargin}%</span>
              </div>
              <input
                type="range"
                min="40"
                max="80"
                value={opexMargin}
                onChange={(e) => setOpexMargin(Number(e.target.value))}
                className="w-full accent-accentCyan"
              />
            </div>

            {/* World Cup Boost Settings */}
            <div className="space-y-2 flex items-center justify-between py-1 border-t border-[#1f293d] pt-3">
              <span className="text-xs font-bold text-slate-400 uppercase">Coupe du Monde 2030</span>
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
                  <span className="font-bold text-accentGreen uppercase">Boost ADR Target</span>
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
          </div>

          <button
            onClick={handleRunSimulation}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-accentAmber text-[#090d16] font-bold py-2.5 rounded-xl hover:bg-accentAmber/90 disabled:opacity-50 transition-all duration-200 shadow-[0_0_15px_rgba(255,179,0,0.2)]"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-t-transparent border-[#090d16] rounded-full animate-spin"></div>
            ) : (
              <>
                <Dices size={16} />
                <span>Simuler les Risques</span>
              </>
            )}
          </button>
        </div>

        {/* Right Side: Simulation Results Visualizations */}
        <div className="lg:col-span-3 space-y-8">
          {!result && !loading && (
            <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center text-center space-y-4">
              <div className="p-4 bg-accentAmber/10 text-accentAmber rounded-full">
                <Dices size={36} />
              </div>
              <h3 className="text-lg font-bold text-white">Simulation Prête à être Lancée</h3>
              <p className="text-slate-400 max-w-md text-sm">
                Réglez les bornes d'incertitude dans le panneau de gauche et lancez les tirages de Monte Carlo pour analyser la distribution de probabilité du retour sur investissement.
              </p>
            </div>
          )}

          {loading && (
            <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center space-y-4">
              <div className="w-12 h-12 border-4 border-t-accentAmber border-slate-700 rounded-full animate-spin"></div>
              <h3 className="text-md font-bold text-white">Calcul probabiliste en cours...</h3>
              <p className="text-slate-400 text-sm max-w-sm text-center">
                Modélisation de distributions normales autour de vos paramètres. Résolution des VAN et TRI pour chaque tirage stochastique.
              </p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Risk metrics grids */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Metric 1 */}
                <div className="glass-panel p-5 rounded-2xl relative overflow-hidden">
                  <span className="text-xs font-bold text-slate-400 uppercase">VAN Attendue (Moyenne)</span>
                  <div className="mt-2 flex items-baseline justify-between">
                    <span className="text-2xl font-extrabold text-white">
                      {result.summary.Expected_NPV_MUSD.toFixed(2)} M$
                    </span>
                    <span className="text-xs text-slate-400">σ = {result.summary.Std_NPV_MUSD.toFixed(1)}M$</span>
                  </div>
                </div>

                {/* Metric 2 */}
                <div className="glass-panel p-5 rounded-2xl relative overflow-hidden border-l-4 border-accentRose">
                  <span className="text-xs font-bold text-slate-400 uppercase">Value at Risk (VaR 95%)</span>
                  <div className="mt-2 flex items-baseline justify-between">
                    <span className={`text-2xl font-extrabold ${result.summary.ValueAtRisk_95_MUSD >= 0 ? 'text-accentGreen' : 'text-accentRose'}`}>
                      {result.summary.ValueAtRisk_95_MUSD.toFixed(2)} M$
                    </span>
                    <span className="text-[10px] text-slate-400">NPV Min (Confidence 95%)</span>
                  </div>
                </div>

                {/* Metric 3 */}
                <div className="glass-panel p-5 rounded-2xl relative overflow-hidden">
                  <span className="text-xs font-bold text-slate-400 uppercase">Probabilité de Perte</span>
                  <div className="mt-2 flex items-baseline justify-between">
                    <span className={`text-2xl font-extrabold ${result.summary.ProbabilityOfLoss_Pct > 15 ? 'text-accentRose' : 'text-accentGreen'}`}>
                      {result.summary.ProbabilityOfLoss_Pct.toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-slate-400">Prob(NPV &lt; 0)</span>
                  </div>
                </div>

                {/* Metric 4 */}
                <div className="glass-panel p-5 rounded-2xl relative overflow-hidden">
                  <span className="text-xs font-bold text-slate-400 uppercase">TRI Attendu</span>
                  <div className="mt-2 flex items-baseline justify-between">
                    <span className="text-2xl font-extrabold text-accentGreen">
                      {result.summary.Expected_IRR_Pct.toFixed(2)}%
                    </span>
                    <span className="text-[10px] text-slate-400">Moyenne Stochastique</span>
                  </div>
                </div>
              </div>

              {/* Histogram Plot */}
              <div className="glass-panel p-6 rounded-2xl space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-white">Distribution de Probabilité du ROI Cumulé</h3>
                  <p className="text-xs text-slate-400 mt-0.5">Fréquence de distribution du retour sur investissement sur les {result.summary.SimulationsCount} scénarios simulés</p>
                </div>

                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.histogram} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                      <XAxis dataKey="BinRange" stroke="#475569" fontSize={10} tickLine={false} />
                      <YAxis stroke="#475569" fontSize={11} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#121824', borderColor: '#1f293d', borderRadius: '12px' }}
                        formatter={(value) => [value, 'Fréquence (Tirages)']}
                      />
                      <Bar dataKey="Count" fill="#ffb300" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Technical explanations card */}
                <div className="bg-[#121824] border border-[#1f293d] rounded-xl p-4 flex gap-3 text-xs text-slate-300">
                  <Info size={24} className="text-accentCyan shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <p className="font-bold text-white">Interprétation Analytique</p>
                    <p className="leading-relaxed">
                      L'intervalle de confiance à 90% pour le ROI se situe entre <span className="font-bold text-accentCyan">{result.summary.CI_ROI_Low.toFixed(1)}%</span> (5ème pct) et <span className="font-bold text-accentCyan">{result.summary.CI_ROI_High.toFixed(1)}%</span> (95ème pct). La Value at Risk (VaR 95%) de <span className="font-bold text-accentRose">{result.summary.ValueAtRisk_95_MUSD.toFixed(2)} M$</span> indique que dans le pire des scénarios (5% de probabilité), la VAN ne descendra pas en dessous de ce seuil.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default MonteCarlo
