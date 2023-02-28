import './App.css';
import Navbar from './components/navbar/Navbar';
import Hero from './components/hero/Hero';
import Features from './components/features/Features';
import Tools from './components/tools/Tools';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <Features />
      <Tools />
    </div>
  );
}

export default App;
