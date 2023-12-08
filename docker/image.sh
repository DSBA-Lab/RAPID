image_name=bb451/log:cuml_1.13.1_11.6
user_password=3775
echo "Image_name: " $image_name
echo "User_password: " $user_password

docker build -t $image_name --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg user_password=$user_password .