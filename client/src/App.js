import {BrowserRouter, Routes, Route} from "react-router-dom"
import Home from "./components/Home"
import ChatPage from "./components/ChatPage";

// connect to where the Dr Cloud is project running and emitting the ws connection
// const socket = new WebSocket("ws://localhost:8000/dr_claude");
const socket = new WebSocket("ws://...../dr_claude");


socket.onopen = () => {
  console.log("WebSocket connection established.");

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
};

socket.onclose = (event) => {
  console.log("WebSocket connection closed:", event.code, event.reason);
};

function App() {
  return (
    <BrowserRouter>
        <div>
          <Routes>
            <Route path="/" element={<Home socket={socket}/>}></Route>
            <Route path="/chat" element={<ChatPage socket={socket}/>}></Route>
          </Routes>
    </div>
    </BrowserRouter>
    
  );
}

export default App;
