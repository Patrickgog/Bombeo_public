@echo off
chcp 65001 >nul
title Upload to GitHub - Gestor de Archivos

echo.
echo ========================================
echo    UPLOAD TO GITHUB - GESTOR DE ARCHIVOS
echo ========================================
echo.

:verificar_configuracion
echo Verificando configuración de Git...
git remote -v >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ¡CONFIGURACIÓN INICIAL REQUERIDA!
    echo.
    goto configurar_github
)

:menu_principal
echo Selecciona una opción:
echo.
echo 1. Subir archivo específico
echo 2. Subir todos los archivos modificados
echo 3. Ver estado actual del repositorio
echo 4. Configurar cuenta de GitHub
echo 5. Cambiar repositorio de destino
echo 6. Salir
echo.
set /p opcion="Ingresa el número de opción: "

if "%opcion%"=="1" goto subir_archivo
if "%opcion%"=="2" goto subir_todos
if "%opcion%"=="3" goto ver_estado
if "%opcion%"=="4" goto configurar_github
if "%opcion%"=="5" goto cambiar_repositorio
if "%opcion%"=="6" goto salir
echo Opción inválida. Intenta de nuevo.
goto menu_principal

:configurar_github
echo.
echo ========================================
echo       CONFIGURACIÓN DE GITHUB
echo ========================================
echo.
echo Configurando Git para GitHub...
echo.

set /p username="Ingresa tu nombre de usuario de GitHub: "
set /p email="Ingresa tu email de GitHub: "
set /p token="Ingresa tu Personal Access Token (o presiona Enter si no lo tienes): "

if "%token%"=="" (
    echo.
    echo Para crear un Personal Access Token:
    echo 1. Ve a GitHub.com → Settings → Developer settings
    echo 2. Personal access tokens → Tokens (classic)
    echo 3. Generate new token → Give it a name
    echo 4. Selecciona 'repo' permissions
    echo 5. Copia el token generado
    echo.
    pause
    goto configurar_github
)

echo.
echo Configurando Git...
git config --global user.name "%username%"
git config --global user.email "%email%"

echo.
echo Configuración completada.
echo Usuario: %username%
echo Email: %email%
echo.
pause
goto menu_principal

:cambiar_repositorio
echo.
echo ========================================
echo    CAMBIAR REPOSITORIO DE DESTINO
echo ========================================
echo.
echo Repositorio actual:
git remote -v
echo.

echo Selecciona una opción:
echo.
echo 1. Ingresar URL manualmente
echo 2. Listar repositorios disponibles (requiere GitHub CLI)
echo 3. Volver al menú principal
echo.
set /p opcion_repo="Ingresa el número de opción: "

if "%opcion_repo%"=="1" goto ingresar_url_manual
if "%opcion_repo%"=="2" goto listar_repositorios
if "%opcion_repo%"=="3" goto menu_principal
echo Opción inválida.
goto cambiar_repositorio

:ingresar_url_manual
echo.
set /p nuevo_repo="Ingresa la URL del nuevo repositorio (ej: https://github.com/usuario/repo.git): "
if "%nuevo_repo%"=="" (
    echo No se ingresó ninguna URL.
    pause
    goto cambiar_repositorio
)
goto cambiar_repositorio_destino

:listar_repositorios
echo.
echo Intentando listar repositorios disponibles...
echo (Esto requiere tener GitHub CLI instalado)
echo.
gh repo list --limit 10 2>nul
if %errorlevel% neq 0 (
    echo.
    echo GitHub CLI no está instalado o no estás autenticado.
    echo Para instalar GitHub CLI: https://cli.github.com/
    echo.
    pause
    goto cambiar_repositorio
)

echo.
set /p repo_seleccionado="Ingresa el nombre del repositorio (ej: usuario/repo): "
if "%repo_seleccionado%"=="" (
    echo No se seleccionó ningún repositorio.
    pause
    goto cambiar_repositorio
)

set nuevo_repo=https://github.com/%repo_seleccionado%.git

:cambiar_repositorio_destino
echo.
echo Cambiando repositorio de destino...
git remote remove origin
git remote add origin "%nuevo_repo%"

echo.
echo Repositorio actualizado:
git remote -v
echo.
pause
goto menu_principal

if "%opcion%"=="1" goto subir_archivo
if "%opcion%"=="2" goto subir_todos
if "%opcion%"=="3" goto ver_estado
if "%opcion%"=="4" goto salir
echo Opción inválida. Intenta de nuevo.
goto menu_principal

:subir_archivo
echo.
echo Selecciona el archivo a subir:
echo.
echo 1. Usar explorador de archivos (recomendado)
echo 2. Listar archivos del directorio actual
echo 3. Volver al menú principal
echo.
set /p opcion_archivo="Ingresa el número de opción: "

if "%opcion_archivo%"=="1" goto browse_archivo
if "%opcion_archivo%"=="2" goto listar_archivos
if "%opcion_archivo%"=="3" goto menu_principal
echo Opción inválida. Intenta de nuevo.
goto subir_archivo

:browse_archivo
echo.
echo Abriendo explorador de archivos...
echo Selecciona el archivo que quieres subir y haz clic en "Abrir"
echo.
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $fileBrowser = New-Object System.Windows.Forms.OpenFileDialog; $fileBrowser.Filter = 'Archivos Python (*.py)|*.py|Archivos de texto (*.txt)|*.txt|Archivos Markdown (*.md)|*.md|Archivos JSON (*.json)|*.json|Todos los archivos (*.*)|*.*'; $fileBrowser.Title = 'Selecciona el archivo a subir a GitHub'; $fileBrowser.InitialDirectory = '%cd%'; if($fileBrowser.ShowDialog() -eq 'OK') { $fileBrowser.FileName }" > temp_file.txt
set /p archivo=<temp_file.txt
del temp_file.txt

if "%archivo%"=="" (
    echo No se seleccionó ningún archivo.
    pause
    goto subir_archivo
)

echo Archivo seleccionado: %archivo%
echo.

REM Verificar si el archivo existe
if not exist "%archivo%" (
    echo ERROR: El archivo seleccionado no existe.
    pause
    goto subir_archivo
)

REM Obtener solo el nombre del archivo (sin ruta completa)
for %%f in ("%archivo%") do set nombre_archivo=%%~nxf
echo Nombre del archivo: %nombre_archivo%
echo.

set /p confirmar_archivo="¿Confirmar este archivo? (s/n): "
if /i "%confirmar_archivo%"=="s" (
    set archivo="%nombre_archivo%"
    goto obtener_mensaje
)
if /i "%confirmar_archivo%"=="n" goto subir_archivo
echo Opción inválida.
goto subir_archivo

:listar_archivos
echo.
echo Archivos disponibles en el directorio:
echo.
dir /b *.py *.txt *.md *.json *.yaml *.yml 2>nul
echo.
set /p archivo="Ingresa el nombre del archivo a subir: "
if not exist "%archivo%" (
    echo ERROR: El archivo "%archivo%" no existe.
    pause
    goto subir_archivo
)
goto obtener_mensaje

:subir_todos
echo.
echo Subiendo todos los archivos modificados...
set archivo="."
goto obtener_mensaje

:obtener_mensaje
echo.
echo ========================================
echo           INFORMACIÓN DEL COMMIT
echo ========================================
echo.

:verificar_primer_commit
git log --oneline -1 >nul 2>&1
if %errorlevel% neq 0 (
    echo ¡PRIMER COMMIT DETECTADO!
    echo.
    set /p version="Ingresa la versión del programa (ej: v1.0.0): "
    set /p mensaje="Ingresa el mensaje del commit: "
    set mensaje_completo="Primer commit: %mensaje% - Versión %version%"
) else (
    echo.
    set /p version="Ingresa la versión del programa (ej: v1.0.1): "
    set /p mensaje="Ingresa el mensaje del commit: "
    set mensaje_completo="%mensaje% - Versión %version%"
)

echo.
echo ========================================
echo           RESUMEN DE CAMBIOS
echo ========================================
echo.
echo Archivo(s): %archivo%
echo Versión: %version%
echo Mensaje: %mensaje%
echo.
set /p confirmar="¿Confirmar y subir a GitHub? (s/n): "

if /i "%confirmar%"=="s" goto ejecutar_upload
if /i "%confirmar%"=="n" goto menu_principal
echo Opción inválida.
goto obtener_mensaje

:ejecutar_upload
echo.
echo ========================================
echo           SUBIENDO A GITHUB
echo ========================================
echo.

echo 1. Agregando archivos...
echo Archivo a agregar: %archivo%

REM Verificar si el archivo existe en el directorio actual
if "%archivo%"=="." (
    echo Agregando todos los archivos modificados...
    git add .
) else (
    echo Verificando si el archivo existe...
    if not exist "%archivo%" (
        echo ERROR: El archivo "%archivo%" no existe en el directorio actual.
        echo Archivos disponibles en el directorio:
        dir /b *.py *.txt *.md *.json *.yaml *.yml 2>nul
        echo.
        pause
        goto menu_principal
    )
    echo Agregando archivo específico...
    git add "%archivo%"
)

if %errorlevel% neq 0 (
    echo ERROR: No se pudieron agregar los archivos.
    echo Archivo intentado: %archivo%
    pause
    goto menu_principal
)

echo 2. Verificando archivos agregados...
git status --porcelain
echo.

echo 3. Haciendo commit...
git commit -m %mensaje_completo%
if %errorlevel% neq 0 (
    echo ERROR: No se pudo hacer el commit.
    echo Verificando estado actual:
    git status
    pause
    goto menu_principal
)

echo 3. Subiendo a GitHub...
git push
if %errorlevel% neq 0 (
    echo ERROR: No se pudo subir a GitHub.
    echo Verifica tu conexión y credenciales.
    pause
    goto menu_principal
)

echo.
echo ========================================
echo           ¡ÉXITO!
echo ========================================
echo.
echo Archivos subidos exitosamente a GitHub.
echo Versión: %version%
echo Mensaje: %mensaje%
echo.
pause
goto menu_principal

:ver_estado
echo.
echo ========================================
echo           ESTADO DEL REPOSITORIO
echo ========================================
echo.
echo Archivos modificados:
git status --porcelain
echo.
echo Últimos commits:
git log --oneline -5
echo.
pause
goto menu_principal

:salir
echo.
echo ¡Hasta luego!
echo.
pause
exit 