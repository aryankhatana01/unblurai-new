import React from 'react'
import './Card.css'
import { Link } from 'react-router-dom'

const Card = (props) => {
  return (
    <div className='card'>
        <Link to={props.link}>
            <div className="card__img">
                <video className='video' muted="" loop="" playsinline="">
                    <source src={props.img} type="video/webm" />
                </video>
            </div>
            <div className="card__title">
                <svg width="1.5rem" height="1.5rem" viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12.5001 12.5H37.5001V37.5H12.5001V12.5ZM12.5001 4.16669H16.6667V10.4167H12.5001V4.16669ZM12.5001 39.5834H16.6667V45.8334H12.5001V39.5834ZM4.16675 12.5H10.4167V16.6667H4.16675V12.5ZM4.16675 33.3334H10.4167V37.5H4.16675V33.3334ZM39.5834 12.5H45.8334V16.6667H39.5834V12.5ZM39.5834 33.3334H45.8334V37.5H39.5834V33.3334ZM33.3334 4.16669H37.5001V10.4167H33.3334V4.16669ZM33.3334 39.5834H37.5001V45.8334H33.3334V39.5834Z" fill="currentcolor"></path></svg>
                <p>{props.title}</p>
            </div>
            <p className='card__desc'>{props.desc}</p>
            <div className="card__btn">
                {props.btn_txt}
            </div>
        </Link>
    </div>
  )
}

export default Card