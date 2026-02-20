import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, ReferenceLine, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, ZAxis
} from 'recharts';
import {
  Activity, Wind, Thermometer, MapPin, RefreshCw, 
  Calendar, Sun, Cloud, AlertCircle, CheckCircle,
  TrendingUp, Droplets, Gauge, Brain, Award, Zap,
  Target, BarChart3, Cpu, Layers, Server, Database,
  ChevronDown, ChevronUp, Info, Download
} from 'lucide-react';

// ==========================================
// 1. CONFIGURATION
// ==========================================
const API_BASE_URL = 'http://localhost:8000';

// ==========================================
// 2. UI COMPONENTS
// ==========================================

const Card = ({ title, icon: Icon, children, className = "", loading = false }) => (
  <div className={`bg-gray-900 border border-gray-800 rounded-xl p-6 ${className}`}>
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-sm font-semibold text-gray-400 flex items-center gap-2">
        {Icon && <Icon size={16} className="text-blue-500" />}
        {title}
      </h3>
      {loading && (
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      )}
    </div>
    <div style={{ minHeight: '200px', width: '100%' }}>
      {children}
    </div>
  </div>
);

const StatCard = ({ label, value, unit, icon: Icon, color = "blue", trend = null }) => {
  const colorClasses = {
    blue: "text-blue-500 bg-blue-500/10",
    green: "text-green-500 bg-green-500/10",
    yellow: "text-yellow-500 bg-yellow-500/10",
    red: "text-red-500 bg-red-500/10",
    purple: "text-purple-500 bg-purple-500/10"
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500 mb-1">{label}</p>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-white">{value}</span>
            <span className="text-sm text-gray-500">{unit}</span>
          </div>
          {trend && (
            <p className={`text-xs mt-1 ${trend > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from yesterday
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon size={20} />
        </div>
      </div>
    </div>
  );
};

const AQIGauge = ({ value }) => {
  const getColor = (aqi) => {
    if (aqi <= 50) return '#10b981';
    if (aqi <= 100) return '#fbbf24';
    if (aqi <= 200) return '#f97316';
    if (aqi <= 300) return '#ef4444';
    if (aqi <= 400) return '#a855f7';
    return '#7f1d1d';
  };

  const getCategory = (aqi) => {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Satisfactory';
    if (aqi <= 200) return 'Moderate';
    if (aqi <= 300) return 'Poor';
    if (aqi <= 400) return 'Very Poor';
    return 'Severe';
  };

  const percentage = Math.min((value / 500) * 100, 100);
  const color = getColor(value);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="#1f2937"
            strokeWidth="10"
          />
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={`${percentage * 2.83} 283`}
            strokeDashoffset="0"
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-white">{Math.round(value)}</span>
          <span className="text-xs text-gray-500">AQI</span>
        </div>
      </div>
      <span className="mt-2 text-sm font-medium" style={{ color }}>{getCategory(value)}</span>
    </div>
  );
};

const ModelMetricCard = ({ model, isSelected, onSelect, isBest }) => (
  <div 
    onClick={() => onSelect(model.id)}
    className={`cursor-pointer p-4 rounded-xl border-2 transition-all relative ${
      isSelected 
        ? 'border-blue-500 bg-blue-500/10' 
        : 'border-gray-800 bg-gray-900 hover:border-gray-700'
    }`}
  >
    {isBest && !isSelected && (
      <div className="absolute -top-2 -right-2 bg-yellow-500 text-black text-[10px] font-bold px-2 py-0.5 rounded-full uppercase flex items-center gap-1">
        <Award size={10} /> Best
      </div>
    )}
    <div className="flex items-center gap-2 mb-3">
      <div className={`w-3 h-3 rounded-full ${isSelected ? 'bg-blue-500 shadow-[0_0_10px_#3b82f6]' : 'bg-gray-600'}`}></div>
      <span className={`font-bold text-sm ${isSelected ? 'text-white' : 'text-gray-400'}`}>{model.name}</span>
      <span className="text-[10px] px-2 py-0.5 bg-gray-800 rounded-full text-gray-500 ml-auto">{model.type}</span>
    </div>
    
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-xs text-gray-500">Accuracy</span>
        <span className={`text-sm font-mono ${isSelected ? 'text-blue-400' : 'text-gray-400'}`}>
          {model.accuracy}%
        </span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5">
        <div 
          className="bg-blue-500 h-1.5 rounded-full" 
          style={{ width: `${model.accuracy}%` }}
        ></div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
        <div>
          <p className="text-gray-600">Precision</p>
          <p className="text-white font-mono">{model.precision}%</p>
        </div>
        <div>
          <p className="text-gray-600">Recall</p>
          <p className="text-white font-mono">{model.recall}%</p>
        </div>
        <div>
          <p className="text-gray-600">F1 Score</p>
          <p className="text-white font-mono">{model.f1}%</p>
        </div>
        <div>
          <p className="text-gray-600">RMSE</p>
          <p className="text-white font-mono">{model.rmse}</p>
        </div>
      </div>
    </div>
  </div>
);

// ==========================================
// 3. MAIN DASHBOARD
// ==========================================

export default function AQIDashboard() {
  // State
  const [forecast, setForecast] = useState([]);
  const [latestData, setLatestData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDay, setSelectedDay] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [selectedModelId, setSelectedModelId] = useState('xgb');
  const [showModelDetails, setShowModelDetails] = useState(true);
  const [modelMetrics, setModelMetrics] = useState(null);

  // Mock model data for visualization (will be replaced by API data)
  const [models, setModels] = useState([
    { 
      id: 'xgb', 
      name: 'XGBoost', 
      type: 'Boosting', 
      accuracy: 97.2, 
      precision: 96.5, 
      recall: 95.8, 
      f1: 96.1, 
      rmse: 9.4, 
      mae: 6.2, 
      latency: 45, 
      training: 120, 
      color: '#10b981',
      features: 14,
      cvScore: 96.8,
      importance: { pm25: 0.28, pm10: 0.22, no2: 0.15, so2: 0.12, o3: 0.11, co: 0.07, temp: 0.05 }
    },
    { 
      id: 'lgbm', 
      name: 'LightGBM', 
      type: 'Boosting', 
      accuracy: 96.8, 
      precision: 96.0, 
      recall: 95.5, 
      f1: 95.7, 
      rmse: 10.1, 
      mae: 6.8, 
      latency: 35, 
      training: 90, 
      color: '#3b82f6',
      features: 14,
      cvScore: 96.2,
      importance: { pm25: 0.26, pm10: 0.21, no2: 0.16, so2: 0.13, o3: 0.12, co: 0.08, temp: 0.04 }
    },
    { 
      id: 'rf', 
      name: 'Random Forest', 
      type: 'Ensemble', 
      accuracy: 94.5, 
      precision: 93.8, 
      recall: 93.2, 
      f1: 93.5, 
      rmse: 13.5, 
      mae: 9.5, 
      latency: 120, 
      training: 180, 
      color: '#8b5cf6',
      features: 14,
      cvScore: 93.9,
      importance: { pm25: 0.24, pm10: 0.20, no2: 0.17, so2: 0.14, o3: 0.13, co: 0.09, temp: 0.03 }
    },
    { 
      id: 'cat', 
      name: 'CatBoost', 
      type: 'Boosting', 
      accuracy: 96.9, 
      precision: 96.2, 
      recall: 95.9, 
      f1: 96.0, 
      rmse: 9.8, 
      mae: 6.5, 
      latency: 55, 
      training: 110, 
      color: '#f59e0b',
      features: 14,
      cvScore: 96.4,
      importance: { pm25: 0.27, pm10: 0.22, no2: 0.15, so2: 0.12, o3: 0.11, co: 0.08, temp: 0.05 }
    },
    { 
      id: 'nn', 
      name: 'Neural Net', 
      type: 'Deep Learning', 
      accuracy: 92.1, 
      precision: 91.5, 
      recall: 90.8, 
      f1: 91.1, 
      rmse: 18.2, 
      mae: 12.5, 
      latency: 15, 
      training: 450, 
      color: '#ec4899',
      features: 14,
      cvScore: 91.5,
      importance: { pm25: 0.23, pm10: 0.19, no2: 0.18, so2: 0.15, o3: 0.14, co: 0.08, temp: 0.03 }
    }
  ]);

  const bestModelId = useMemo(() => {
    return models.sort((a, b) => b.accuracy - a.accuracy)[0].id;
  }, [models]);

  const selectedModel = models.find(m => m.id === selectedModelId) || models[0];

  // Check backend connection
  const checkBackend = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus('connected');
        return true;
      }
    } catch (err) {
      console.log('Backend not reachable');
    }
    setBackendStatus('disconnected');
    return false;
  };

  // Fetch model metrics
  const fetchModelMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/model-metrics`);
      if (response.ok) {
        const data = await response.json();
        setModels(data.models);
        setSelectedModelId(data.selected_model);
        setModelMetrics(data);
      }
    } catch (err) {
      console.log('Model metrics not available, using defaults');
    }
  };

  // Fetch data from backend
  const fetchData = async () => {
    try {
      setRefreshing(true);
      
      // First check if backend is reachable
      const isBackendReachable = await checkBackend();
      
      if (!isBackendReachable) {
        throw new Error('Backend not reachable');
      }
      
      // Fetch forecast
      const forecastRes = await fetch(`${API_BASE_URL}/api/forecast?days=3`);
      if (!forecastRes.ok) throw new Error('Failed to fetch forecast');
      const forecastData = await forecastRes.json();
      
      // Fetch latest data
      const latestRes = await fetch(`${API_BASE_URL}/api/latest`);
      const latestData = await latestRes.json();
      
      // Fetch model metrics
      await fetchModelMetrics();
      
      setForecast(forecastData.forecast);
      setLatestData(latestData);
      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to connect to backend. Please make sure the server is running on port 8000.');
      
      // Generate mock data as fallback
      generateMockData();
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Generate mock data for fallback
  const generateMockData = () => {
    const mockForecast = [];
    const now = new Date();
    
    for (let i = 0; i < 72; i++) {
      const time = new Date(now.getTime() + i * 3600000);
      const hour = time.getHours();
      const baseAqi = 120 + 40 * Math.sin(hour / 12 * Math.PI) + Math.random() * 15;
      
      mockForecast.push({
        time: time.toISOString(),
        aqi: Math.max(0, Math.min(500, baseAqi)),
        temperature: 25 + 5 * Math.sin(hour / 12 * Math.PI) + Math.random() * 2,
        weather: hour > 6 && hour < 18 ? 'Sunny' : 'Clear'
      });
    }
    
    setForecast(mockForecast);
    setLatestData({
      time: now.toISOString(),
      pm25: 45.2,
      aqi: 112,
      category: 'Moderate',
      temperature: 28.5,
      humidity: 65,
      pm10: 78,
      no2: 35
    });
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Process data for charts
  const chartData = useMemo(() => {
    return forecast.map(item => ({
      ...item,
      hour: new Date(item.time).getHours(),
      day: Math.floor((new Date(item.time) - new Date()) / (24 * 3600000)),
      timeLabel: new Date(item.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      dayLabel: new Date(item.time).toLocaleDateString([], { weekday: 'short' })
    }));
  }, [forecast]);

  // Group by day
  const dailyForecast = useMemo(() => {
    const groups = {};
    chartData.forEach(item => {
      const dayKey = new Date(item.time).toDateString();
      if (!groups[dayKey]) {
        groups[dayKey] = [];
      }
      groups[dayKey].push(item);
    });
    
    return Object.entries(groups).map(([day, data], index) => ({
      day: index === 0 ? 'Today' : index === 1 ? 'Tomorrow' : new Date(day).toLocaleDateString([], { weekday: 'long' }),
      data,
      avgAqi: Math.round(data.reduce((sum, d) => sum + d.aqi, 0) / data.length),
      maxAqi: Math.max(...data.map(d => d.aqi)),
      minAqi: Math.min(...data.map(d => d.aqi))
    }));
  }, [chartData]);

  // Get current day's data
  const currentDayData = dailyForecast[selectedDay]?.data || [];

  // Feature importance data for chart
  const featureImportanceData = useMemo(() => {
    if (!selectedModel.importance) return [];
    return Object.entries(selectedModel.importance).map(([name, value]) => ({
      name: name.toUpperCase(),
      value: value * 100
    }));
  }, [selectedModel]);

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 text-blue-500 animate-pulse mx-auto mb-4" />
          <p className="text-gray-400">Loading dashboard...</p>
          <p className="text-xs text-gray-600 mt-2">Make sure backend is running on port 8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <Activity className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">AQI Forecast Dashboard</h1>
              <p className="text-xs text-gray-500 flex items-center gap-1">
                <MapPin size={12} /> Karachi, Pakistan
                {latestData && (
                  <>
                    <span className="mx-2">•</span>
                    <span>Updated: {new Date(latestData.time).toLocaleTimeString()}</span>
                  </>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Backend status indicator */}
            <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
              backendStatus === 'connected' ? 'bg-green-500/10 text-green-500' : 
              backendStatus === 'disconnected' ? 'bg-red-500/10 text-red-500' : 
              'bg-yellow-500/10 text-yellow-500'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                backendStatus === 'connected' ? 'bg-green-500 animate-pulse' : 
                backendStatus === 'disconnected' ? 'bg-red-500' : 
                'bg-yellow-500'
              }`}></div>
              <span>{backendStatus === 'connected' ? 'Live' : backendStatus === 'disconnected' ? 'Offline' : 'Checking...'}</span>
            </div>
            <button
              onClick={fetchData}
              disabled={refreshing}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition flex items-center gap-2"
            >
              <RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />
              <span className="text-sm hidden sm:inline">Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 mt-4">
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-center gap-3">
            <AlertCircle className="text-red-500 w-5 h-5 flex-shrink-0" />
            <p className="text-sm text-red-500">{error}</p>
            <button 
              onClick={fetchData}
              className="ml-auto text-xs bg-red-500/20 hover:bg-red-500/30 text-red-500 px-3 py-1 rounded"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-6">
        {/* Current Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            label="Current AQI"
            value={latestData?.aqi?.toFixed(0) || '112'}
            unit=""
            icon={Gauge}
            color={latestData?.aqi <= 100 ? 'green' : latestData?.aqi <= 200 ? 'yellow' : 'red'}
          />
          <StatCard
            label="PM2.5"
            value={latestData?.pm25?.toFixed(1) || '45.2'}
            unit="µg/m³"
            icon={Wind}
            color="blue"
          />
          <StatCard
            label="Temperature"
            value={latestData?.temperature?.toFixed(1) || '28.5'}
            unit="°C"
            icon={Thermometer}
            color="yellow"
          />
          <StatCard
            label="Humidity"
            value={latestData?.humidity?.toFixed(0) || '65'}
            unit="%"
            icon={Droplets}
            color="purple"
          />
        </div>

        {/* Main Gauge & Quick Stats */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card title="Air Quality Index" icon={Activity}>
            <AQIGauge value={latestData?.aqi || 112} />
            <div className="mt-4 grid grid-cols-2 gap-2 text-center text-sm">
              <div className="bg-gray-800 rounded-lg p-2">
                <p className="text-gray-500">PM10</p>
                <p className="text-white font-bold">{latestData?.pm10?.toFixed(0) || '78'}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-2">
                <p className="text-gray-500">NO2</p>
                <p className="text-white font-bold">{latestData?.no2?.toFixed(0) || '35'}</p>
              </div>
            </div>
          </Card>

          <Card title="Today's Forecast" icon={Calendar} className="lg:col-span-2">
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={currentDayData.slice(0, 24)}>
                  <defs>
                    <linearGradient id="aqiGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis 
                    dataKey="hour" 
                    tickFormatter={(h) => `${h}:00`}
                    stroke="#4b5563"
                  />
                  <YAxis stroke="#4b5563" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="aqi" 
                    stroke="#3b82f6" 
                    fill="url(#aqiGradient)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* 3-Day Forecast Overview */}
        <Card title="3-Day Forecast" icon={Calendar}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {dailyForecast.slice(0, 3).map((day, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedDay(idx)}
                className={`p-4 rounded-lg border-2 transition ${
                  selectedDay === idx 
                    ? 'border-blue-500 bg-blue-500/10' 
                    : 'border-gray-800 hover:border-gray-700'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="font-bold text-white">{day.day}</span>
                  <span className="text-sm text-gray-500">{day.data.length}h</span>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-white">{day.avgAqi}</p>
                    <p className="text-xs text-gray-500">Avg AQI</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">
                      <span className="text-red-500">{Math.round(day.maxAqi)}</span>
                      <span className="text-gray-600 mx-1">/</span>
                      <span className="text-green-500">{Math.round(day.minAqi)}</span>
                    </p>
                    <p className="text-xs text-gray-500">Max/Min</p>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {/* Detailed Hourly Chart */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={currentDayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis 
                  dataKey="hour" 
                  tickFormatter={(h) => `${h}:00`}
                  interval={3}
                  stroke="#4b5563"
                />
                <YAxis yAxisId="left" stroke="#4b5563" />
                <YAxis yAxisId="right" orientation="right" stroke="#4b5563" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                  labelStyle={{ color: '#9ca3af' }}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="aqi" fill="#3b82f6" name="AQI" />
                <Line 
                  yAxisId="right" 
                  type="monotone" 
                  dataKey="temperature" 
                  stroke="#f97316" 
                  name="Temperature °C"
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Hourly Weather Cards */}
        <Card title="Next 12 Hours" icon={Sun}>
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-12 gap-2">
            {currentDayData.slice(0, 12).map((hour, idx) => (
              <div key={idx} className="bg-gray-800 rounded-lg p-2 text-center">
                <p className="text-xs text-gray-500">{hour.timeLabel}</p>
                {hour.weather?.toLowerCase().includes('sun') ? (
                  <Sun className="w-5 h-5 text-yellow-500 mx-auto my-1" />
                ) : (
                  <Cloud className="w-5 h-5 text-gray-400 mx-auto my-1" />
                )}
                <p className="text-sm font-bold text-white">{Math.round(hour.aqi)}</p>
                <p className="text-xs text-gray-500">{Math.round(hour.temperature)}°</p>
              </div>
            ))}
          </div>
        </Card>

        {/* ========== MODEL METRICS SECTION ========== */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-white font-bold flex items-center gap-2">
              <Brain size={20} className="text-blue-500" /> Model Performance Metrics
            </h2>
            <button
              onClick={() => setShowModelDetails(!showModelDetails)}
              className="text-xs text-gray-500 hover:text-white flex items-center gap-1"
            >
              {showModelDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {showModelDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          {/* Model Selection Cards */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {models.map((model) => (
              <ModelMetricCard
                key={model.id}
                model={model}
                isSelected={selectedModelId === model.id}
                isBest={model.id === bestModelId}
                onSelect={setSelectedModelId}
              />
            ))}
          </div>

          {showModelDetails && (
            <>
              {/* Model Comparison Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                {/* Radar Chart - Model Comparison */}
                <Card title="Model Performance Comparison" icon={Target}>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={models.map(m => ({ name: m.name, accuracy: m.accuracy }))}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                        <PolarRadiusAxis angle={45} domain={[80, 100]} tick={{ fill: '#6b7280' }} />
                        {models.map(m => (
                          <Radar 
                            key={m.id}
                            name={m.name} 
                            dataKey="accuracy" 
                            stroke={m.color} 
                            fill={m.color} 
                            fillOpacity={m.id === selectedModelId ? 0.4 : 0.05} 
                            strokeWidth={m.id === selectedModelId ? 3 : 1}
                          />
                        ))}
                        <Legend />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {/* Feature Importance */}
                <Card title={`Feature Importance - ${selectedModel.name}`} icon={BarChart3}>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportanceData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis type="number" domain={[0, 30]} stroke="#9ca3af" />
                        <YAxis type="category" dataKey="name" stroke="#9ca3af" width={60} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                          formatter={(value) => [`${value.toFixed(1)}%`, 'Importance']}
                        />
                        <Bar dataKey="value" fill={selectedModel.color} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {/* Error Analysis */}
                <Card title="Error Analysis (RMSE vs MAE)" icon={Zap}>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis type="number" dataKey="rmse" name="RMSE" stroke="#9ca3af" />
                        <YAxis type="number" dataKey="mae" name="MAE" stroke="#9ca3af" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                          formatter={(value, name) => [value, name]}
                        />
                        {models.map(m => (
                          <Scatter 
                            key={m.id}
                            name={m.name}
                            data={[m]}
                            fill={m.color}
                            shape={m.id === selectedModelId ? "star" : "circle"}
                            isAnimationActive={false}
                          />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-gray-500 text-center mt-2">Lower values are better (bottom-left)</p>
                  </div>
                </Card>

                {/* Latency vs Accuracy */}
                <Card title="Efficiency Analysis" icon={Cpu}>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis type="number" dataKey="latency" name="Latency (ms)" stroke="#9ca3af" />
                        <YAxis type="number" dataKey="accuracy" name="Accuracy %" domain={[90, 100]} stroke="#9ca3af" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                          formatter={(value, name) => [value, name === 'accuracy' ? 'Accuracy %' : 'Latency (ms)']}
                        />
                        {models.map(m => (
                          <Scatter 
                            key={m.id}
                            name={m.name}
                            data={[m]}
                            fill={m.color}
                            shape={m.id === selectedModelId ? "star" : "circle"}
                            isAnimationActive={false}
                          />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-gray-500 text-center mt-2">Ideal: Top-left (high accuracy, low latency)</p>
                  </div>
                </Card>
              </div>

              {/* Detailed Metrics Table */}
              <Card title="Detailed Metrics Matrix" icon={Layers}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-gray-500 uppercase border-b border-gray-800">
                      <tr>
                        <th className="py-3 px-4">Metric</th>
                        {models.map(m => (
                          <th key={m.id} className={`py-3 px-4 font-semibold ${m.id === selectedModelId ? 'text-blue-400' : 'text-gray-400'}`}>
                            {m.name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="text-gray-300">
                      {[
                        { key: 'accuracy', label: 'Accuracy (%)' },
                        { key: 'precision', label: 'Precision (%)' },
                        { key: 'recall', label: 'Recall (%)' },
                        { key: 'f1', label: 'F1 Score (%)' },
                        { key: 'cvScore', label: 'CV Score (%)' },
                        { key: 'rmse', label: 'RMSE' },
                        { key: 'mae', label: 'MAE' },
                        { key: 'latency', label: 'Latency (ms)' },
                        { key: 'training', label: 'Training Time (s)' },
                        { key: 'features', label: 'Features' }
                      ].map(metric => {
                        const values = models.map(m => m[metric.key]);
                        const bestValue = metric.key.includes('rmse') || metric.key.includes('mae') || metric.key.includes('latency') || metric.key.includes('training')
                          ? Math.min(...values)
                          : Math.max(...values);
                        
                        return (
                          <tr key={metric.key} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                            <td className="py-3 px-4 font-medium text-gray-400">{metric.label}</td>
                            {models.map(m => {
                              const isBest = m[metric.key] === bestValue;
                              const isSelected = m.id === selectedModelId;
                              return (
                                <td key={m.id} className={`py-3 px-4 font-mono ${isSelected ? 'text-white font-bold' : ''}`}>
                                  <span className={isBest ? 'text-green-500' : ''}>
                                    {typeof m[metric.key] === 'number' ? m[metric.key].toFixed(1) : m[metric.key]}
                                  </span>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Model Summary */}
              <Card title="Model Summary" icon={Info}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-3">Selected Model: {selectedModel.name}</h4>
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Type</p>
                        <p className="text-sm text-white">{selectedModel.type}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Training Samples</p>
                        <p className="text-sm text-white">10,000 records</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Cross-validation Score</p>
                        <p className="text-sm text-white">{selectedModel.cvScore}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Last Trained</p>
                        <p className="text-sm text-white">{new Date().toLocaleDateString()}</p>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-3">Top Features</h4>
                    <div className="space-y-2">
                      {featureImportanceData.slice(0, 5).map((feat, idx) => (
                        <div key={feat.name}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-400">{feat.name}</span>
                            <span className="text-white">{feat.value.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-1.5">
                            <div 
                              className="bg-blue-500 h-1.5 rounded-full" 
                              style={{ width: `${feat.value}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            </>
          )}
        </div>

        {/* Status Footer */}
        <div className="flex items-center justify-between text-xs text-gray-600 border-t border-gray-800 pt-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              {backendStatus === 'connected' ? (
                <CheckCircle size={12} className="text-green-500" />
              ) : (
                <AlertCircle size={12} className="text-red-500" />
              )}
              <span>Backend {backendStatus}</span>
            </div>
            <div className="flex items-center gap-1">
              <Cpu size={12} className="text-blue-500" />
              <span>Active Model: {selectedModel.name}</span>
            </div>
          </div>
          <p>Last updated: {new Date().toLocaleString()}</p>
        </div>
      </main>
    </div>
  );
}