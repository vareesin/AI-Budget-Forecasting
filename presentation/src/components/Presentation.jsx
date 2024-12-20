import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const modelSampleData = {
  lstm: Array.from({ length: 20 }, (_, i) => ({
    name: i,
    actual: Math.sin(i * 0.5) * 10 + 20,
    predicted: Math.sin(i * 0.5) * 10 + 20 + Math.random()
  }))
};

const slides = [
  {
    title: "AI for Budgets Forecasting",
    subtitle: "AI-Driven Forecasting Models",
    description: "A comprehensive AI system for improving budget forecasting accuracy and efficiency",
    type: "title"
  },
  {
    title: "Project Overview",
    points: [
      "AI system for budget forecasting",
      "Aims to solve manual data preparation issues",
      "Reduces forecasting inaccuracies",
      "Enables strategic focus for employees",
      "Implements multiple AI models for optimal results"
    ],
    type: "points"
  },
  {
    title: "PAIN POINT",
    points: [
      "Store Coordinators manually prepare sales data",
      "Delays in forecasting process",
      "Manual forecasting by FCs",
      "Inaccurate forecasts and inefficient allocation",
      "Resources spent on data handling"
    ],
    type: "points"
  },
  {
    title: "SOLUTION",
    subtitle: "AI-Driven Forecasting Models",
    points: [
      "LSTM",
      "GRU",
      "CNN-LSTM",
      "XGBoost",
      "Prophet",
      "Random Forest"
    ],
    libraries: [
      "Pandas",
      "Matplotlib",
      "Scikit-learn",
      "TensorFlow"
    ],
    type: "solution"
  },
  {
    title: "Model Performance",
    type: "chart",
  },
  {
    title: "Process Steps",
    points: [
      "1. Data Preparation",
      "2. Model Selection",
      "3. Outlier Detection",
      "4. Prediction Generation",
      "5. Results Validation"
    ],
    type: "points"
  },
  {
    title: "Results",
    traditional: {
      title: "Traditional Method",
      accuracy: "80%",
      cost: "1,781,615 THB/year"
    },
    ai: {
      title: "AI-Driven Method",
      accuracy: "84%",
      savings: "Full automation"
    },
    type: "results"
  },
  {
    title: "Technical Implementation",
    sections: [
      {
        title: "Data Preparation",
        points: [
          "Load 2023-2024 profit data",
          "Clean and merge multiple files",
          "Standardize dates and column names"
        ]
      },
      {
        title: "Model Training",
        points: [
          "80% training data",
          "20% testing data",
          "6-month future forecasting"
        ]
      }
    ],
    type: "technical"
  },
  {
    title: "Key Factors",
    points: [
      "Campaign performance",
      "Promotional activities",
      "Historical profit data",
      "Seasonal patterns",
      "Market trends"
    ],
    type: "points"
  },
  {
    title: "Cost Savings",
    savings: [
      {
        title: "Store Coordinators",
        amount: "37,050 THB/month",
        detail: "5 minutes saved per branch"
      },
      {
        title: "Financial Controllers",
        amount: "111,418 THB/month",
        detail: "10 minutes saved per branch"
      }
    ],
    type: "savings"
  }
];

const ModelChart = () => (
  <div className="w-full h-48">
    <ResponsiveContainer>
      <LineChart data={modelSampleData.lstm}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" dot={false} />
        <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const Slide = ({ slide }) => {
  switch (slide.type) {
    case 'title':
      return (
        <div className="text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h1>
          <p className="text-xl text-orange-500">
            {slide.subtitle}
          </p>
          <div className="mt-4 text-sm text-gray-500">
          <h2>Varees Adulyasas - Group K</h2>
          <h1>Creative AI Camp 2024</h1>
          </div>
        </div>
      );

    case 'points':
      return (
        <div>
          <h2 className="text-3xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h2>
          <ul className="space-y-2">
            {slide.points.map((point, idx) => (
              <li key={idx} className="text-base">• {point}</li>
            ))}
          </ul>
        </div>
      );

    case 'solution':
      return (
        <div>
          <h2 className="text-3xl font-bold text-blue-600 mb-2">
            {slide.title}
          </h2>
          <p className="text-lg mb-4">{slide.subtitle}</p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold mb-2">Models:</h3>
              <ul className="space-y-1">
                {slide.points.map((point, idx) => (
                  <li key={idx} className="text-sm">• {point}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Libraries:</h3>
              <ul className="space-y-1">
                {slide.libraries.map((lib, idx) => (
                  <li key={idx} className="text-sm">• {lib}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      );

    case 'chart':
      return (
        <div className="w-full">
          <h2 className="text-3xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h2>
          <ModelChart />
        </div>
      );

    case 'results':
      return (
        <div>
          <h2 className="text-3xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h2>
          <div className="grid grid-cols-2 gap-8">
            <div className="border p-4 rounded-lg">
              <h3 className="font-semibold mb-2">{slide.traditional.title}</h3>
              <p>Accuracy: {slide.traditional.accuracy}</p>
              <p>Cost: {slide.traditional.cost}</p>
            </div>
            <div className="border p-4 rounded-lg">
              <h3 className="font-semibold mb-2">{slide.ai.title}</h3>
              <p>Accuracy: {slide.ai.accuracy}</p>
              <p>Cost: {slide.ai.savings}</p>
            </div>
          </div>
        </div>
      );

    case 'technical':
      return (
        <div>
          <h2 className="text-3xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h2>
          <div className="space-y-6">
            {slide.sections.map((section, idx) => (
              <div key={idx} className="space-y-2">
                <h3 className="font-semibold text-xl">{section.title}</h3>
                <ul className="list-disc pl-5 space-y-1">
                  {section.points.map((point, i) => (
                    <li key={i} className="text-sm">{point}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      );

    case 'savings':
      return (
        <div>
          <h2 className="text-3xl font-bold text-blue-600 mb-4">
            {slide.title}
          </h2>
          <div className="grid grid-cols-2 gap-4">
            {slide.savings.map((item, idx) => (
              <div key={idx} className="border p-4 rounded-lg">
                <h3 className="font-semibold mb-2">{item.title}</h3>
                <p className="text-lg text-green-600">{item.amount}</p>
                <p className="text-sm text-gray-600">{item.detail}</p>
              </div>
            ))}
          </div>
        </div>
      );

    default:
      return null;
  }
};

const Presentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl aspect-[16/9] relative overflow-hidden">
        <div className="absolute top-2 right-2 text-sm text-gray-500">
          {currentSlide + 1} / {slides.length}
        </div>
        
        <div className="h-full flex items-center justify-center p-6">
          <Slide slide={slides[currentSlide]} />
        </div>

        <div className="absolute bottom-2 left-0 right-0 flex justify-center space-x-2">
          <button
            onClick={() => setCurrentSlide(prev => Math.max(0, prev - 1))}
            className="p-1 rounded-full bg-gray-200 hover:bg-gray-300"
            disabled={currentSlide === 0}
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentSlide(prev => Math.min(slides.length - 1, prev + 1))}
            className="p-1 rounded-full bg-gray-200 hover:bg-gray-300"
            disabled={currentSlide === slides.length - 1}
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Presentation;