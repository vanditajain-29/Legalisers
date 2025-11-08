import React, { useState } from "react";
import UploadBox from "./components/UploadBox";
import ChatBox from "./components/ChatBox";
import "../src/styles.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [uploaded, setUploaded] = useState(false);

  const handleUploadSuccess = (msg) => {
    setMessages([{ sender: "bot", text: msg }]);
    setUploaded(true);
  };

  const handleNewMessage = (msg) => {
    setMessages((prev) => [...prev, msg]);
  };

  return (
    <div className="container">
      <h1>⚖️ Legal Document Analyzer</h1>
      <p className="subtitle">Analyze and understand your documents</p>

      <UploadBox onUploadSuccess={handleUploadSuccess} />

      {uploaded && (
        <ChatBox messages={messages} setMessages={setMessages} onNewMessage={handleNewMessage} />
      )}
    </div>
  );
}

export default App;
