import React, { useState } from "react";
import axios from "axios";
import MessageBubble from "./MessageBubble";

const ChatBox = ({ messages, setMessages, onNewMessage }) => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const BACKEND_URL = "http://127.0.0.1:8000";

  const sendMessage = async () => {
    if (!query.trim()) return;
    const newMsg = { sender: "user", text: query };
    setMessages((prev) => [...prev, newMsg]);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post(`${BACKEND_URL}/chat`, { query });
      const botReply = res.data.answer || "No response.";
      onNewMessage({ sender: "bot", text: botReply });
    } catch {
      onNewMessage({ sender: "bot", text: "⚠️ Error processing your question." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-box">
      <div className="messages">
        {messages.map((msg, i) => (
          <MessageBubble key={i} sender={msg.sender} text={msg.text} />
        ))}
      </div>

      <div className="input-row">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about this document..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default ChatBox;
