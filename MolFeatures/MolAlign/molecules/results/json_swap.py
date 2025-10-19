import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Listbox, Scrollbar
import json
import os

class JSONRowSwapper(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JSON Row Swapper")
        self.geometry("600x400")
        self.json_data = None
        self.json_path = None
        self.selected_entry_idx = None
        self.selected_key = None

        # GUI Elements
        self.load_btn = tk.Button(self, text="Load JSON File", command=self.load_json)
        self.load_btn.pack(pady=10)

        self.entries_listbox = Listbox(self, width=80, height=10)
        self.entries_listbox.pack()
        self.entries_listbox.bind('<<ListboxSelect>>', self.on_entry_select)

        self.keys_listbox = Listbox(self, width=40, height=5)
        self.keys_listbox.pack()
        self.keys_listbox.bind('<<ListboxSelect>>', self.on_key_select)

        self.rows_listbox = Listbox(self, width=80, height=10)
        self.rows_listbox.pack()

        self.swap_btn = tk.Button(self, text="Swap Rows", command=self.swap_rows, state=tk.DISABLED)
        self.swap_btn.pack(pady=10)

        self.save_btn = tk.Button(self, text="Save JSON File", command=self.save_json, state=tk.DISABLED)
        self.save_btn.pack(pady=10)

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not path:
            return
        with open(path, 'r') as f:
            self.json_data = json.load(f)
        self.json_path = path
        self.entries_listbox.delete(0, tk.END)
        for idx, entry in enumerate(self.json_data):
            self.entries_listbox.insert(tk.END, f"Entry {idx}: {str(entry)[:80]}")
        self.keys_listbox.delete(0, tk.END)
        self.rows_listbox.delete(0, tk.END)
        self.swap_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)

    def on_entry_select(self, event):
        selection = self.entries_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.selected_entry_idx = idx
        entry = self.json_data[idx]
        self.keys_listbox.delete(0, tk.END)
        self.rows_listbox.delete(0, tk.END)
        if isinstance(entry, dict):
            for key in entry.keys():
                self.keys_listbox.insert(tk.END, key)
        else:
            self.display_rows(entry)
            self.selected_key = None
    def on_key_select(self, event):
        selection = self.keys_listbox.curselection()
        if not selection:
            return
        key = self.keys_listbox.get(selection[0])
        self.selected_key = key
        entry = self.json_data[self.selected_entry_idx]
        rows = entry[key]
        if isinstance(rows, list):
            self.display_rows(rows)
        else:
            self.rows_listbox.delete(0, tk.END)
            self.rows_listbox.insert(tk.END, f"Value for key '{key}' is not a list: {rows}")
            self.swap_btn.config(state=tk.DISABLED)

    def display_rows(self, rows):
        self.rows_listbox.delete(0, tk.END)
        if not isinstance(rows, list):
            self.rows_listbox.insert(tk.END, f"Cannot display: not a list ({type(rows)})")
            self.swap_btn.config(state=tk.DISABLED)
            return
        for idx, row in enumerate(rows):
            self.rows_listbox.insert(tk.END, f"Row {idx}: {row}")
        self.swap_btn.config(state=tk.NORMAL)



    def swap_rows(self):
        selected = self.rows_listbox.curselection()
        if len(selected) != 2:
            messagebox.showerror("Error", "Please select exactly two rows to swap.")
            return
        i, j = selected
        if self.selected_key is not None:
            rows = self.json_data[self.selected_entry_idx][self.selected_key]
        else:
            rows = self.json_data[self.selected_entry_idx]
        rows[i], rows[j] = rows[j], rows[i]
        self.display_rows(rows)
        messagebox.showinfo("Success", f"Swapped rows {i} and {j}.")
        self.save_btn.config(state=tk.NORMAL)

    def save_json(self):
        if not self.json_path:
            self.json_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not self.json_path:
            return
        with open(self.json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)
        messagebox.showinfo("Saved", f"JSON file saved to {self.json_path}")

if __name__ == "__main__":
    app = JSONRowSwapper()
    app.mainloop()
