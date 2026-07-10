@echo off
rem Plan B: dispara la actualizacion del SAT en GitHub cada 6 horas.
rem Instalado como tarea programada "SAT Cali Plan B" (ver README).
rem Ruta completa de python: el Programador de tareas no hereda el PATH de Anaconda.
cd /d "D:\Proyectos Personales\SAT Incendios\SAT_Cali\scripts"
"C:\Users\jairo\anaconda3\python.exe" dispatch_workflow.py >> ..\plan_b.log 2>&1
