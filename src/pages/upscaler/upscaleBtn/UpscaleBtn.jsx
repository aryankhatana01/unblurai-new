import React, { useContext } from "react";
import './UpscaleBtn.css';
import FileContext from '../contexts/FileContext'


const UpscaleBtn = () => {
    const selectedFile = useContext(FileContext);
    // const [pred, setPred] = useState(null);

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append("file", selectedFile, selectedFile.name);
        console.log(formData);
        console.log(selectedFile.name);

        const requestOptions = {
            method: 'POST',
            body: formData,
        };
        await fetch('http://localhost:8000/uploadfile', requestOptions);
        // const data = await resp.json();
    }
    const handlePredict = async () => {
        await handleUpload();
        // const requestOptions = {
        //     method: 'GET',
        // };
        // const resp = await fetch(`http://localhost:8000/predict/?filename=${selectedFile.name}`, requestOptions);
        // const data = await resp.json();
        // setPred(data["Prediction"]);
        // console.log(data);
        // console.log(pred);
    }
    return (
        <div className="predict-button">
            <button className="predict-button-button" onClick={handlePredict}>Upscale!</button>
        </div>
    )
}

export default UpscaleBtn;