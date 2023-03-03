import './App.css';
import Navbar from './components/navbar/Navbar';
import HomePage from './pages/homepage/HomePage';
import { Routes, Route } from 'react-router-dom';
import Upscaler from './pages/upscaler/Upscaler';
import Relight from './pages/relight/Relight';
import RemoveBG from './pages/removebg/RemoveBG';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/upscaler" element={<Upscaler />} />
        <Route path="/relight" element={<Relight />} />
        <Route path="/removebg" element={<RemoveBG />} />
      </Routes>
    </div>
  );
}

export default App;
