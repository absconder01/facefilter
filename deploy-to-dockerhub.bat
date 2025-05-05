@echo off
setlocal

REM Kullanıcı bilgileri
set DOCKER_USERNAME=absconder
set IMAGE_NAME=facefilter-dlib-app
set TAG=latest

set FULL_IMAGE_NAME=%DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%

echo 🔧 Docker imajı build ediliyor: %FULL_IMAGE_NAME%
docker build -t %FULL_IMAGE_NAME% .

IF ERRORLEVEL 1 (
    echo ❌ Build sırasında hata oluştu
    exit /b 1
)

echo 🔐 Docker Hub'a giriş yapılıyor...
docker login

IF ERRORLEVEL 1 (
    echo ❌ Giriş başarısız
    exit /b 1
)

echo 📤 Docker Hub'a push ediliyor: %FULL_IMAGE_NAME%
docker push %FULL_IMAGE_NAME%

IF ERRORLEVEL 1 (
    echo ❌ Push işlemi başarısız oldu.
    exit /b 1
) ELSE (
    echo ✅ Başarılı: %FULL_IMAGE_NAME% Docker Hub'a yüklendi.
    echo 🌐 Render.com üzerinde bu image'ı kullanabilirsin.
)

endlocal
