import React from 'react';
import { Cloud, Database, Shield, RefreshCw, AlertTriangle } from 'lucide-react';

const WeatherSourceInfo = ({ data, }) => {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-4 mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <Cloud className="w-5 h-5 text-blue-600" />
          </div>
          <div>
            <h3 className="font-bold text-blue-900">OpenWeather API</h3>
            <p className="text-sm text-blue-600">Live weather & pollution data source</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 text-sm">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-green-700 font-medium">Live</span>
          </div>
          <RefreshCw className="w-4 h-4 text-blue-500" />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="bg-white/70 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Database className="w-4 h-4 text-blue-500" />
            <span className="font-medium text-blue-800">Data Sources</span>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-blue-600">Weather API</span>
              <span className="font-medium">✓ Active</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-600">Pollution API</span>
              <span className="font-medium">✓ Active</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-600">Forecast API</span>
              <span className="font-medium">✓ Active</span>
            </div>
          </div>
        </div>
        
        <div className="bg-white/70 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className="w-4 h-4 text-green-500" />
            <span className="font-medium text-green-800">AQI Standards</span>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-green-600">Indian CPCB</span>
              <span className="font-medium">Applied</span>
            </div>
            <div className="flex justify-between">
              <span className="text-green-600">OpenWeather AQI</span>
              <span className="font-medium">Available</span>
            </div>
            <div className="flex justify-between">
              <span className="text-green-600">WHO Guidelines</span>
              <span className="font-medium">Referenced</span>
            </div>
          </div>
        </div>
        
        <div className="bg-white/70 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span className="font-medium text-amber-800">Pollutants Tracked</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            {['PM2.5', 'PM10', 'NO₂', 'SO₂', 'O₃', 'CO', 'NH₃'].map((poll, idx) => (
              <div key={idx} className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-amber-400 rounded-full"></div>
                <span className="text-xs text-amber-700">{poll}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {data?.source === 'database' && (
        <div className="mt-3 pt-3 border-t border-blue-200">
          <div className="text-xs text-blue-600 flex justify-between">
            <span>Last fetched from OpenWeather: {new Date(data.timestamp).toLocaleTimeString()}</span>
            <span className="font-medium">API v2.5</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default WeatherSourceInfo;