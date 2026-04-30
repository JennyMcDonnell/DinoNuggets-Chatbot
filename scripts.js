async function sendMessage() {
    const inputField = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const question = inputField.value.trim();

    //don't do anything if the user entered nothing
    if (!question) return;

    // Display the users message
    addMessage(question, 'user-message');
    // Clear input
    inputField.value = '';


    try {
        // send question to flask
        const response = await fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ "question": question })
        });

        const data = await response.json();

        // display response
        if (data.response) {
            addMessage(data.response, 'bot-message');
        } else {
            addMessage("Oops! DinoNuggets tripped over a bone. Try again.", 'bot-message');
        }
    } catch (error) {
        console.error("Error:", error);
        addMessage("Can't connect to the Dino-Server. Is it running away from Velociraptors?", 'bot-message');
    }
}

//show the message that the user typed back to them
function addMessage(text, className) {
    const chatBox = document.getElementById('chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);

    // Auto-scroll to the bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Allow "Enter" key to send the message
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});