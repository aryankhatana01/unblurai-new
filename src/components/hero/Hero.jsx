import React from 'react'
import './Hero.css'

const Hero = () => {
  return (
    <div className='hero-section'>
        <h1 className='main-heading'>
            Upscale your <span className='images-text'>images</span> by 4x in <span className='just-one-click'>just one click</span>
        </h1>
        <p className='subheading'>Increase the size of your image using AI for completely free.</p>
        <div className="btn-container">
            <a className='try-btn' href="/home">
                Try Now!
            </a>
        </div>
        <div className="section-division-line"></div>
    </div>
  )
}

export default Hero