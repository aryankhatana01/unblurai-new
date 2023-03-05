import React, { useRef } from 'react'
import './UploadSection.css'
import UpscaleBtn from '../upscaleBtn/UpscaleBtn'

const UploadSection = (props) => {
    const fileInputRef = useRef(null);

    const handleButtonClick = () => {
        fileInputRef.current.click();
    }

    const handleFileChange = (e) => {
        props.setSelectedFile(e.target.files[0]);
        console.log(e.target.files[0]);
    }
  return (
    <>
        <div className="drag-n-drop" onClick={handleButtonClick}>
            <input type="file" onChange={handleFileChange} name="file" ref={fileInputRef} style={{display: "none"}}/>
            Click or Drag & Drop an SD Image
        </div>
        <div className="button-upscale">
            <UpscaleBtn />
        </div>
    </>
  )
}

export default UploadSection