import React from "react";

const MessageBubble = ({ sender, text }) => {
  return (
    <div className={`message ${sender}`}>
      <div className="bubble">{text}</div>
    </div>
  );
};

export default MessageBubble;
