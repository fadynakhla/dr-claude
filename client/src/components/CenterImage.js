import React from "react";
import backgroundImage from "../doctor-patient.jpeg";
import bubble from "../bubble.png"
const generateRandomKey = () => Math.random().toString(36).substr(2, 10);

const CenterImage = ({ lastMessageRef, patientMessages, doctorMessages,   lastPatientMessageRef, lastDoctorMessageRef, brainMessages, conditonMessages}) => {
  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          width: "100%",
          height: "100%",
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: "cover",
          backgroundPosition: "center 20%",
          transform: "translate(-50%, -50%)",
          zIndex: "-1",
          padding: "20px",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "14%",
          left: "10%",
          transform: "translate(-50%, -50%)",
          padding: "20px",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "20%",
            left: "22%",
            background: "rgba(255, 255, 255, 0.7)",
            padding: "20px",
            borderRadius: "10px",
            maxWidth: "220px",
            maxHeight: "180px",
            overflow: "auto",
          }}
        >
          {/* Brain Messages*/}
          <div>
            {brainMessages.map((message) =>
              message === localStorage.getItem("userName") ? (
                <div className="message__chats" key={generateRandomKey()} ref={lastPatientMessageRef}>
                  {/* <p className='sender__name'>You</p> */}
                  <div>
                    <p style={{ fontSize: "18px" }}>{message}</p>
                  </div>
                </div>
              ) : (
                <div className="message__chats" key={generateRandomKey()}>
                  {/* <p>{message.name}</p> */}
                  <div c>
                    <p style={{ fontSize: "18px" }}>{message}</p>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
        <img src={bubble} style={{ width: "266px", height: "266px" }} alt="My Image" />
      </div>
      <div
        style={{
          position: "absolute",
          top: "14%",
          left: "40%",
          transform: "translate(-50%, -50%)",
          background: "rgba(255, 255, 255, 0.7)",
          padding: "20px",
          borderRadius: "10px",
          maxWidth: "450px",
          maxHeight: "240px",
          overflow: "auto",
        }}
      >
        <div>
          {doctorMessages.map((message) =>
            message === localStorage.getItem("userName") ? (
              <div className="message__chats" key={generateRandomKey()}>
                {/* <p className='sender__name'>You</p> */}
                <div className="message__sender">
                  <p>{message}</p>
                </div>
              </div>
            ) : (
              <div className="message__chats" key={generateRandomKey()}>
                {/* <p>{message.name}</p> */}
                <div className="message__sender">
                  <p>{message}</p>
                </div>
              </div>
            )
          )}
        </div>
        <div ref={lastDoctorMessageRef} />
      </div>
      <div
        style={{
          position: "fixed",
          top: "25%",
          right: "8%",
        //   background: "rgba(255, 255, 255, 0.7)",
          padding: "15px",
          borderRadius: "2px",
          maxWidth: "400px",
          maxHeight: "190px",
          overflow: "auto",
        }}
      >
        <div>
          {patientMessages.map((message) =>
            message === localStorage.getItem("userName") ? (
              <div className="message__chats" key={generateRandomKey()}>
                {/* <p className='sender__name'>You</p> */}
                <div className="message__sender">
                  <p>{message}</p>
                </div>
              </div>
            ) : (
              <div className="message__chats" key={generateRandomKey()}>
                {/* <p>{message.name}</p> */}
                <div className="message__recipient">
                  <p>{message}</p>
                </div>
              </div>
            )
          )}
        </div>
        <div ref={lastPatientMessageRef} />
      </div>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "53%",
          transform: "translate(-50%, -50%)",
        //   background: "rgba(255, 255, 255, 0.7)",
          padding: "20px",
          borderRadius: "10px",
          maxWidth: "450px",
          maxHeight: "215px",
          overflow: "auto",
        }}
      >
        {/* Disease - Condition */}
        
        <div>
          {conditonMessages.map((message) =>
            message === localStorage.getItem("userName") ? (
              <div className="message__chats" key={generateRandomKey()}>
                <div className="message__sender">
                  <p>{`Looks like you have ${message}`}</p>
                </div>
              </div>
            ) : (
              <div className="message__chats" key={generateRandomKey()}>
                <div className="message__recipient">
                  <p>{`Looks like you have ${message}`}</p>
                </div>
              </div>
            )
          )}
        </div>
        
        <div ref={lastMessageRef} />
      </div>
    </div>
  );
};

export default CenterImage;