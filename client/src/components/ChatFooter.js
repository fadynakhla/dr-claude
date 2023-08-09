import React, {useState} from 'react'

const ChatFooter = ({socket}) => {
    const [message, setMessage] = useState("")

    const handleSendMessage = (e) => {
        e.preventDefault()
        const messageToSend = {
          type: "message",
          content: message
        };
        
        socket.send(JSON.stringify(messageToSend));
        setMessage("")
    }
  return (
    <div className='chat__footer'>
        <form className='form' onSubmit={handleSendMessage}>
          <input 
            type="text" 
            placeholder='Write message' 
            className='message' 
            value={message} 
            onChange={e => setMessage(e.target.value)}
            />
            <button className="sendBtn">SEND</button>
        </form>
     </div>
  )
}

export default ChatFooter