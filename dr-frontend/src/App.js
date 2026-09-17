import React, { useState, useRef } from 'react';

export default function DRDetectionDashboard() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ml');
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Reference to the file input so we can reset it
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  // NEW: Function to clear the selected image and reset results
  const handleClearImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setPrediction(null);
    setConfidence(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Resets the actual file input element
    }
  };

  const handlePredict = async () => {
    if (!imageFile) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("model_type", selectedModel);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction from server");
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence(data.confidence);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8 font-sans">
      <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg p-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Diabetic Retinopathy Detection
        </h1>

        {/* 1. Image Upload Section */}
        <div className="mb-8 border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          
          {/* Show the input ONLY if no image is selected */}
          {!imagePreview && (
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange} 
              ref={fileInputRef}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mx-auto"
            />
          )}

          {/* Show the image and the X button if an image IS selected */}
          {imagePreview && (
            <div className="relative inline-block mt-4">
              <img 
                src={imagePreview} 
                alt="Retina Scan Preview" 
                className="mx-auto max-h-64 object-contain rounded-md shadow-sm border border-gray-200"
              />
              
              {/* The Red Cross Button */}
              <button
                onClick={handleClearImage}
                className="absolute -top-3 -right-3 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-red-600 shadow-md font-bold transition-colors"
                title="Remove Image"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* 2. Model Selection Buttons */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-700 mb-4 text-center">Select Detection Engine</h3>
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => setSelectedModel('ml')}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                selectedModel === 'ml' 
                ? 'bg-blue-600 text-white shadow-md' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Machine Learning (RF)
            </button>
            <button
              onClick={() => setSelectedModel('dl')}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                selectedModel === 'dl' 
                ? 'bg-purple-600 text-white shadow-md' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Deep Learning (CNN)
            </button>
            <button
              onClick={() => setSelectedModel('transformer')}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                selectedModel === 'transformer' 
                ? 'bg-emerald-600 text-white shadow-md' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Transformer (ViT)
            </button>
          </div>
        </div>

        {/* 3. Run Prediction Button */}
        <div className="text-center mb-8">
          <button 
            onClick={handlePredict}
            disabled={!imageFile || loading}
            className="w-full max-w-md bg-gray-900 text-white font-bold py-4 rounded-lg hover:bg-gray-800 disabled:bg-gray-400 transition-colors"
          >
            {loading ? 'Analyzing Scan...' : 'Run Diagnostics'}
          </button>
        </div>

        {/* 4. Results Display */}
        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-6 text-center">
            {error}
          </div>
        )}

        {prediction && !loading && (
          <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg text-center shadow-inner">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Diagnosis: {prediction}</h2>
            <p className="text-gray-600">
              Confidence Score: <span className="font-semibold text-blue-700">{(confidence * 100).toFixed(2)}%</span>
            </p>
            <p className="text-sm text-gray-500 mt-4">
              Engine used: <span className="uppercase font-semibold">{selectedModel}</span>
            </p>
          </div>
        )}

      </div>
    </div>
  );
}