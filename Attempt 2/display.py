# display.py
import tkinter as tk

class PlaceholderEntry(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color="grey", *args, **kwargs):
        kwargs.setdefault("borderwidth", 1)
        kwargs.setdefault("relief", "solid")
        super().__init__(master, *args, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self.cget("fg")
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)
        self._add_placeholder()

    def _clear_placeholder(self, event):
        if self.get() == self.placeholder and self.cget("fg") == self.placeholder_color:
            self.delete(0, tk.END)
            self.config(fg=self.default_fg_color)

    def _add_placeholder(self, event=None):
        if not self.get():
            self.insert(0, self.placeholder)
            self.config(fg=self.placeholder_color)
