# Cara installasi dan running File flask

- setelah di ekstrak file nya, jalankan file di command prompt.
- hidupkan virtual env nya terlebih dahulu.

  ```bash
  .venv\Scripts\activate
    ```
- kalau laptop nya membutuhkan izin/security, beri dulu perizinan di CMD.

    ```bash
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ```
    ```bash
  Unblock-File -Path ".venv\Scripts\Activate.ps1"
    ```
     ```bash
  .venv\Scripts\activate
    ```
- Jika sudah keluar .venv nya, selanjutnya install 'requirements.txt'
    ```bash
   pip install -r requirements.txt
    ```
- Selanjutnya jika sudah selesai installasi, jalankan file Flask nya :
   ```bash
   python wsgi.py
    ```
- notes : port bisa di ubah sesuai keinginan.

## Cara imigration database 
- notes : jika db kosong
- Langkah pertama :

  ```bash
  db_init = "flask db init"
  db_migrate = "flask db migrate"
  db_upgrade = "flask db upgrade"
    ```