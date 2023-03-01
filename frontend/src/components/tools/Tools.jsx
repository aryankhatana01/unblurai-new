import React from 'react'
import './Tools.css'
import Card from '../tools-card/Card'

const Tools = () => {
  return (
    <>
      <div className='heading'>Tools</div>
      <div className='tools'>
        <Card 
          img="https://static.clipdrop.co/web/homepage/tools/Enhance.webm#t=0.1"
          title="Image Upscaler"
          desc="Upscale your images by 4x in just one click. It can also remove noise and recover details."
          btn_txt="Upscale"
        />
        <Card
          img="https://static.clipdrop.co/web/homepage/tools/RemoveBG.webm#t=0.1"
          title="Background Remover"
          desc="Extract the main subject from a picture with incredible accuracy. It's like magic."
          btn_txt="Remove BG"
        />
        <Card 
          img="https://static.clipdrop.co/web/homepage/tools/Relight.webm#t=0.1"
          title="Image Relighting"
          desc="Change the lighting of your images to make them look more natural. It's like magic."
          btn_txt="Relight"
        />
      </div>
    </>
  )
}

export default Tools