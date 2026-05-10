import { useState } from 'react'
import './App.css'

function App() {
  const [formData, setFormData] = useState({
    steps_daily: '12000',
    sleep_minutes: '420',
    time_in_bed: '460',
    sleep_efficiency: '95',
    hr_mean: '66',
    hr_min: '48',
    hr_max: '130',
    hr_std: '14',
    resting_heart_rate: '54',
    calories_daily: '3400',
    mood: '3',
    model_type: 'llm',
  })

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function handleChange(e) {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!formData.steps_daily || !formData.sleep_minutes || !formData.mood) {
      setError('Please fill in all required fields.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch('http://localhost:8000/predict/readiness', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error('Please check the input fields.')
      }

      setResult(data)
    } catch (err) {
      setError(typeof err === 'string' ? err : err?.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Readiness Prediction Console</h1>
          <p className="subtitle">
            Prototype for testing daily readiness predictions from wearable summary data.
          </p>
        </div>
      </header>

      <main className="app-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Inputs</h2>
            <p>Enter one day of features and choose an inference backend.</p>
          </div>

          <form className="prediction-form" onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="steps_daily">Daily Steps</label>
                <input
                  id="steps_daily"
                  type="number"
                  name="steps_daily"
                  value={formData.steps_daily}
                  onChange={handleChange}
                  placeholder="e.g. 12000"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="sleep_minutes">Sleep Minutes</label>
                <input
                  id="sleep_minutes"
                  type="number"
                  name="sleep_minutes"
                  value={formData.sleep_minutes}
                  onChange={handleChange}
                  placeholder="e.g. 420"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="time_in_bed">Time in Bed (min)</label>
                <input
                  id="time_in_bed"
                  type="number"
                  name="time_in_bed"
                  value={formData.time_in_bed}
                  onChange={handleChange}
                  placeholder="e.g. 450"
                />
              </div>

              <div className="field">
                <label htmlFor="sleep_efficiency">Sleep Efficiency (%)</label>
                <input
                  id="sleep_efficiency"
                  type="number"
                  name="sleep_efficiency"
                  value={formData.sleep_efficiency}
                  onChange={handleChange}
                  placeholder="e.g. 93.3"
                />
              </div>

              <div className="field">
                <label htmlFor="hr_mean">Mean Heart Rate</label>
                <input
                  id="hr_mean"
                  type="number"
                  name="hr_mean"
                  value={formData.hr_mean}
                  onChange={handleChange}
                  placeholder="e.g. 65"
                />
              </div>

              <div className="field">
                <label htmlFor="hr_min">Min Heart Rate</label>
                <input
                  id="hr_min"
                  type="number"
                  name="hr_min"
                  value={formData.hr_min}
                  onChange={handleChange}
                  placeholder="e.g. 50"
                />
              </div>

              <div className="field">
                <label htmlFor="hr_max">Max Heart Rate</label>
                <input
                  id="hr_max"
                  type="number"
                  name="hr_max"
                  value={formData.hr_max}
                  onChange={handleChange}
                  placeholder="e.g. 120"
                />
              </div>
              
              <div className="field">
                <label htmlFor="hr_std">Heart Rate Std Dev</label>
                <input
                  id="hr_std"
                  type="number"
                  name="hr_std"
                  value={formData.hr_std}
                  onChange={handleChange}
                  placeholder="e.g. 8.5"
                />
              </div>

              <div className="field">
                <label htmlFor="resting_heart_rate">Resting Heart Rate</label>
                <input
                  id="resting_heart_rate"
                  type="number"
                  name="resting_heart_rate"
                  value={formData.resting_heart_rate}
                  onChange={handleChange}
                  placeholder="e.g. 60"
                />
              </div>

              <div className="field">
                <label htmlFor="calories_daily">Daily Calories Burned</label>
                <input
                  id="calories_daily"
                  type="number"
                  name="calories_daily"
                  value={formData.calories_daily}
                  onChange={handleChange}
                  placeholder="e.g. 2500"
                />
              </div>

              <div className="field">
                <label htmlFor="mood">Mood</label>
                <input
                  id="mood"
                  type="number"
                  name="mood"
                  value={formData.mood}
                  onChange={handleChange}
                  placeholder="e.g. 3"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="model_type">Model</label>
                <select
                  id="model_type"
                  name="model_type"
                  value={formData.model_type}
                  onChange={handleChange}
                  required
                >
                  <option value="llm">LLM</option>
                  <option value="random_forest">Random Forest</option>
                </select>
              </div>
            </div>

            <button className="predict-button" type="submit" disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </form>
        </section>

        <section className="panel result-panel">
          <div className="panel-header">
            <h2>Prediction</h2>
            <p>Output returned by the API.</p>
          </div>

          {!loading && !error && !result && (
            <div className="empty-state">
              Submit the form to view a readiness prediction.
            </div>
          )}

          {loading && <div className="status loading">Generating prediction...</div>}
          {error && <div className="status error">{error}</div>}

          {result && (
            <div className="result-card">
              <div className="result-score">
                <span className="result-label">Predicted Readiness</span>
                <strong>{result.predicted_readiness ?? 'N/A'}</strong>
              </div>

              <div className="result-meta">
                <div>
                  <span>Model</span>
                  <strong>{result.model_name}</strong>
                </div>
                <div>
                  <span>Prompt Version</span>
                  <strong>{result.prompt_version}</strong>
                </div>
                <div>
                  <span>Parse Success</span>
                  <strong>{String(result.parse_success)}</strong>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
