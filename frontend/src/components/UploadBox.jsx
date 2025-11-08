import React, { useState } from "react";
import axios from "axios";

const UploadBox = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [uploading, setUploading] = useState(false);

  const BACKEND_URL = "http://127.0.0.1:8000";

  const handleUpload = async () => {
    if (!file) return alert("Please choose a file first.");
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", title || file.name);

    try {
      const res = await axios.post(`${BACKEND_URL}/chat_upload`, formData);
      onUploadSuccess(res.data.message);
    } catch (err) {
      alert(err.response?.data?.error || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-box">
      <h3>📄 Upload Document</h3>
      <input
        type="file"
        accept=".pdf,.docx,.txt"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <input
        type="text"
        placeholder="Enter document title (optional)"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <button onClick={handleUpload} disabled={uploading}>
        {uploading ? "Uploading..." : "Upload"}
      </button>
    </div>
  );
};

export default UploadBox;
