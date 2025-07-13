import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import shutil

def run_git_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

def subir_archivo():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    file_name = os.path.basename(file_path)
    if not os.path.exists(file_name):
        shutil.copy(file_path, file_name)
    version = entry_version.get()
    msg = entry_msg.get()
    cmds = [
        f'git add "{file_name}"',
        f'git commit -m "{msg} - Versión {version}"',
        'git push'
    ]
    resp = ''
    for cmd in cmds:
        resp += run_git_command(cmd)
    messagebox.showinfo("Resultado", resp)

def subir_todos():
    version = entry_version2.get()
    msg = entry_msg2.get()
    resp = ''
    # 1. git add .
    resp_add = run_git_command('git add .')
    resp += resp_add
    # 2. git commit
    resp_commit = run_git_command(f'git commit -m "{msg} - Versión {version}"')
    resp += resp_commit
    # Si no hay nada para commitear, mostrar mensaje claro y no hacer push
    if "nothing to commit" in resp_commit or "no changes added to commit" in resp_commit or "nothing added to commit" in resp_commit:
        messagebox.showinfo("Resultado", "No hay cambios nuevos para subir.\n\n" + resp_commit)
        return
    # 3. git push
    resp_push = run_git_command('git push')
    resp += resp_push
    messagebox.showinfo("Resultado", resp)

def ver_estado():
    resp = run_git_command('git status') + '\n' + run_git_command('git log --oneline -5')
    text_estado.delete(1.0, tk.END)
    text_estado.insert(tk.END, resp)

def configurar_git():
    user = entry_user.get()
    email = entry_email.get()
    resp_user = run_git_command(f'git config --global user.name "{user}"')
    resp_email = run_git_command(f'git config --global user.email "{email}"')
    # Verificar si la configuración fue exitosa
    check_user = run_git_command('git config --global user.name').strip()
    check_email = run_git_command('git config --global user.email').strip()
    if check_user == user and check_email == email:
        messagebox.showinfo("Resultado", f"¡Configuración exitosa!\nUsuario: {check_user}\nEmail: {check_email}")
    else:
        messagebox.showerror("Error", "No se pudo configurar Git. Por favor, revisa las credenciales e inténtalo de nuevo.")

def cambiar_repo():
    repo = entry_repo.get()
    resp = run_git_command('git remote remove origin')
    resp += run_git_command(f'git remote add origin {repo}')
    messagebox.showinfo("Resultado", resp)

root = tk.Tk()
root.title("Gestor gráfico de GitHub")
root.geometry("650x400")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Subir archivo
frame_subir = ttk.Frame(notebook)
notebook.add(frame_subir, text='Subir archivo')
ttk.Label(frame_subir, text="Versión:").grid(row=0, column=0, sticky='e')
entry_version = ttk.Entry(frame_subir, width=10)
entry_version.insert(0, "v1.0.0")
entry_version.grid(row=0, column=1, sticky='w')
ttk.Label(frame_subir, text="Mensaje de commit:").grid(row=1, column=0, sticky='e')
entry_msg = ttk.Entry(frame_subir, width=40)
entry_msg.grid(row=1, column=1, sticky='w')
ttk.Button(frame_subir, text="Seleccionar y subir archivo", command=subir_archivo).grid(row=2, column=0, columnspan=2, pady=10)

# Subir todos
frame_todos = ttk.Frame(notebook)
notebook.add(frame_todos, text='Subir todos')
ttk.Label(frame_todos, text="Versión:").grid(row=0, column=0, sticky='e')
entry_version2 = ttk.Entry(frame_todos, width=10)
entry_version2.insert(0, "v1.0.0")
entry_version2.grid(row=0, column=1, sticky='w')
ttk.Label(frame_todos, text="Mensaje de commit:").grid(row=1, column=0, sticky='e')
entry_msg2 = ttk.Entry(frame_todos, width=40)
entry_msg2.grid(row=1, column=1, sticky='w')
ttk.Button(frame_todos, text="Subir todos los archivos modificados", command=subir_todos).grid(row=2, column=0, columnspan=2, pady=10)

# Ver estado
frame_estado = ttk.Frame(notebook)
notebook.add(frame_estado, text='Ver estado')
ttk.Button(frame_estado, text="Ver estado del repositorio", command=ver_estado).pack(pady=5)
text_estado = scrolledtext.ScrolledText(frame_estado, width=80, height=15)
text_estado.pack()

# Configurar GitHub
frame_config = ttk.Frame(notebook)
notebook.add(frame_config, text='Configurar GitHub')
ttk.Label(frame_config, text="Usuario:").grid(row=0, column=0, sticky='e')
entry_user = ttk.Entry(frame_config, width=30)
entry_user.grid(row=0, column=1, sticky='w')
ttk.Label(frame_config, text="Email:").grid(row=1, column=0, sticky='e')
entry_email = ttk.Entry(frame_config, width=30)
entry_email.grid(row=1, column=1, sticky='w')
ttk.Button(frame_config, text="Configurar Git", command=configurar_git).grid(row=2, column=0, columnspan=2, pady=10)

# Cambiar repo
frame_repo = ttk.Frame(notebook)
notebook.add(frame_repo, text='Cambiar repo')
ttk.Label(frame_repo, text="URL del repositorio:").grid(row=0, column=0, sticky='e')
entry_repo = ttk.Entry(frame_repo, width=50)
entry_repo.grid(row=0, column=1, sticky='w')
ttk.Button(frame_repo, text="Cambiar repositorio", command=cambiar_repo).grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()