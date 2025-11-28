import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import re
import random
from datetime import datetime
import webbrowser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
#  AI Intent Classifier
# =========================================================
class IntentClassifier:
    def __init__(self, intents):
        self.intents = intents
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self._train()

    def _train(self):
        texts = []
        labels = []
        for tag, data in self.intents.items():
            for p in data["patterns"]:
                texts.append(p)
                labels.append(tag)

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict_intent(self, text: str, threshold: float = 0.15):
        text = text.strip()
        if not text:
            return None
        X_test = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X_test)[0]
        idx = probs.argmax()
        best_intent = self.model.classes_[idx]
        score = probs[idx]
        if score >= threshold:
            return best_intent
        return None


# =========================================================
#  ChatBot Brain
# =========================================================
class SmartBot:
    def __init__(self):

        self.name = "friend"

        # Simple GK facts (rule based)
        self.knowledge = {
            "capital of india": "The capital of India is New Delhi 🇮🇳",
            "what is the capital of india": "The capital of India is New Delhi 🇮🇳",
            "who invented computer": "Charles Babbage is known as the father of the computer 🧠",
            "invented computer": "Charles Babbage is known as the father of the computer 🧠",
            "largest ocean": "The Pacific Ocean is the largest ocean 🌊",
            "largest ocean in the world": "The Pacific Ocean is the largest ocean 🌊",
            "fastest animal": "The cheetah is the fastest land animal 🐆",
            "speed of light": "The speed of light is about 299,792 km per second ⚡",
            "tallest mountain": "Mount Everest is the tallest mountain 🏔",
            "highest mountain": "Mount Everest is the tallest mountain 🏔",
        }

        # Intents used for general chat
        self.intents = {
            "greeting": {
                "patterns": ["hi", "hello", "hey", "yo", "good morning", "good evening"],
                "responses": [
                    "Hey {name} 😁!",
                    "Hi {name}! How can I help?",
                    "Hello {name}! 😎"
                ]
            },
            "how_are_you": {
                "patterns": ["how are you", "how r u", "are you ok", "how's it going"],
                "responses": [
                    "I'm running at full speed ⚡ How about you?",
                    "I'm good, just living in Python 😄",
                    "Pretty good! Thanks for asking, {name} 😊"
                ]
            },
            "joke": {
                "patterns": ["joke", "tell me a joke", "funny", "make me laugh"],
                "responses": [
                    "Why do programmers prefer dark mode? Because light attracts bugs 🐛",
                    "There are 10 types of people: those who understand binary and those who don't 😂",
                    "Python programmers don't need glasses — they have IDEs 🤓"
                ]
            },
            "motivate": {
                "patterns": ["i am sad", "motivate me", "i feel low"],
                "responses": [
                    "You’re stronger than you think 💪",
                    "Every day is a new chance to shine ✨",
                    "I believe in you, {name}! You got this 🚀"
                ]
            },
            "thanks": {
                "patterns": ["thanks", "thank you", "thx", "tysm"],
                "responses": [
                    "You're welcome, {name}! 😊",
                    "Anytime, {name}.",
                    "Glad to help 🤖✨"
                ]
            },
            "goodbye": {
                "patterns": ["bye", "goodbye", "see you", "exit", "quit"],
                "responses": [
                    "Bye {name}! Come back soon ✨",
                    "See you later 👋",
                    "Goodbye! Stay awesome 🤩"
                ]
            },
        }

        self.classifier = IntentClassifier(self.intents)

    def reply(self, text: str) -> str:
        txt = text.lower().strip()

        # 1) Name detection
        m = re.search(r"(my name is|i am|i'm)\s+([A-Za-z]+)", txt)
        if m:
            self.name = m.group(2).title()
            return f"Nice to meet you, {self.name}! 🤝"

        # 2) Time
        if "time" in txt:
            now = datetime.now().strftime("%H:%M:%S")
            return f"The current time is ⏰ {now}"

        # 3) GK facts (rule based)
        for key, ans in self.knowledge.items():
            if key in txt:
                return ans

        # 4) ML intent
        intent = self.classifier.predict_intent(txt)
        if intent:
            responses = self.intents[intent]["responses"]
            return random.choice(responses).format(name=self.name)

        # 5) Fallback
        return random.choice([
            "Tell me more… 🤔",
            "Interesting! 😄",
            "Wow, really? 😳",
            "That sounds cool, {name}! 😎".format(name=self.name),
        ])


# =========================================================
#  GUI
# =========================================================
class ChatUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.bot = SmartBot()

        root.title("AI Chatbot Project 🤖✨")
        root.geometry("900x700")
        root.configure(bg="#1E1F22")

        # ---------- Chat area ----------
        self.chat = scrolledtext.ScrolledText(
            root,
            bg="#2C2E33",
            fg="white",
            font=("Times New Roman", 16),
            wrap=tk.WORD
        )
        self.chat.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.chat.config(state=tk.DISABLED)

        # ---------- Image buttons row (under chat) ----------
        img_bar = tk.Frame(root, bg="#1E1F22")
        img_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        categories = ["flowers", "scifi", "scenery", "animals", "cute"]
        emojis = ["🌸", "🚀", "🌄", "🐾", "🥰"]

        for cat, emo in zip(categories, emojis):
            tk.Button(
                img_bar,
                text=f"{emo} {cat.title()}",
                font=("Times New Roman", 13),
                bg="#8e44ad",
                fg="white",
                relief=tk.FLAT,
                command=lambda c=cat: self.show_image(c)
            ).pack(side=tk.LEFT, padx=5)

        # ---------- Bottom input bar ----------
        bottom = tk.Frame(root, bg="#1E1F22")
        bottom.pack(fill=tk.X, padx=10, pady=5)

        self.entry = tk.Entry(
            bottom,
            font=("Times New Roman", 15),
            bg="#3A3D42",
            fg="white"
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)

        tk.Button(
            bottom,
            text="Send ▶",
            font=("Times New Roman", 13),
            bg="#4F8CF5",
            fg="white",
            relief=tk.FLAT,
            command=self.send_message
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            bottom,
            text="🧹 Clear",
            font=("Times New Roman", 13),
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            command=self.clear_chat
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            bottom,
            text="💾 Save",
            font=("Times New Roman", 13),
            bg="#2ecc71",
            fg="white",
            relief=tk.FLAT,
            command=self.save_chat
        ).pack(side=tk.LEFT, padx=3)

        # ---------- 5 fixed image URLs per category ----------
        self.image_urls = {
            "flowers": [
                "https://images.pexels.com/photos/36764/marguerite-daisy-beautiful-beauty.jpg",
                "https://images.pexels.com/photos/712876/pexels-photo-712876.jpeg",
                "https://images.pexels.com/photos/931177/pexels-photo-931177.jpeg",
                "https://images.pexels.com/photos/5418832/pexels-photo-5418832.jpeg",
                "https://images.pexels.com/photos/56866/garden-rose-red-pink-56866.jpeg",
            ],
            "scifi":[
            "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg",  
            "https://images.pexels.com/photos/2837009/pexels-photo-2837009.jpeg",  
            "https://images.pexels.com/photos/3549518/pexels-photo-3549518.jpeg",  
            "https://images.pexels.com/photos/847393/pexels-photo-847393.jpeg",  
            "https://images.pexels.com/photos/3888151/pexels-photo-3888151.jpeg",  
            ],
            "scenery": [
                "https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg",
                "https://images.pexels.com/photos/462162/pexels-photo-462162.jpeg",
                "https://images.pexels.com/photos/414171/pexels-photo-414171.jpeg",
                "https://images.pexels.com/photos/2014422/pexels-photo-2014422.jpeg",
                "https://images.pexels.com/photos/3408744/pexels-photo-3408744.jpeg",
            ],
            "animals": [
                "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",  # dog
                "https://images.pexels.com/photos/333083/pexels-photo-333083.jpeg",   # cat
                "https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg",   # puppy
                "https://images.pexels.com/photos/1334591/pexels-photo-1334591.jpeg", # rabbit
                "https://images.pexels.com/photos/46024/pexels-photo-46024.jpeg",     # lion
            ],
            "cute":[
                "https://images.pexels.com/photos/45170/kittens-cat-cat-puppy-rush-45170.jpeg",  # cute kittens
                "https://images.pexels.com/photos/751602/pexels-photo-751602.jpeg",  # puppy
                "https://images.pexels.com/photos/20787/pexels-photo.jpg",  # hamster
                "https://images.pexels.com/photos/181406/pexels-photo-181406.jpeg",  # baby bunny
                "https://images.pexels.com/photos/302280/pexels-photo-302280.jpeg",  # baby cat staring

            ],
        }

        # index to rotate through 5 images per category
        self.image_index = {cat: 0 for cat in self.image_urls.keys()}

        # ---------- Intro message ----------
        self.add_message(
            "Bot",
            "Hello! I'm your smart AI chatbot 🤖\n"
            "Tell me your name to begin 😄\n"
            "Try clicking the image buttons below!"
        )

    # ---------- Chat helpers ----------
    def add_message(self, who: str, msg: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{who}: {msg}\n\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

    def send_message(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.add_message("You", text)
        reply = self.bot.reply(text)
        self.add_message("Bot", reply)

    def clear_chat(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.config(state=tk.DISABLED)

    def save_chat(self):
        content = self.chat.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Save Chat", "Chat is empty!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Save Chat", "Chat saved successfully ✅")

    # ---------- Image logic: ONLINE ONLY, 5 URLs per category ----------
    def show_image(self, category: str):
        urls = self.image_urls.get(category)
        if not urls:
            self.add_message("Bot", "I don't have images for that category yet 😅")
            return

        idx = self.image_index[category]
        url = urls[idx]
        # rotate index
        self.image_index[category] = (idx + 1) % len(urls)

        webbrowser.open(url)
        self.add_message("Bot", f"Opened a {category} image in your browser 🌐")


# =========================================================
#  Run App
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatUI(root)
    root.mainloop()
