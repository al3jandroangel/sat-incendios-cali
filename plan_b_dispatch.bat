@echo off
rem Plan B: cada 6 horas (1) incorpora al modelo los puntos calientes FIRMS
rem nuevos y reentrena si los hay, (2) dispara la actualizacion en GitHub.
rem Instalado como tarea programada "SAT Cali Plan B" (ver README).
rem Ruta completa de python: el Programador de tareas no hereda el PATH de Anaconda.
cd /d "D:\Proyectos Personales\SAT Incendios\SAT_Cali\scripts"
"C:\Users\jairo\anaconda3\python.exe" 09_actualizar_incendios.py --auto >> ..\plan_b.log 2>&1
"C:\Users\jairo\anaconda3\python.exe" dispatch_workflow.py >> ..\plan_b.log 2>&1
