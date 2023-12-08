container_name=logAD
image_name=bb451/log:cuml_1.13.1_11.6

echo "Container_name: " $container_name
echo "Image_name: " $image_name

docker run -td \
	-p 3775:3775 \
    --ipc=host \
    --name $container_name \
	--gpus all \
	-v /ssd2/logAD:/home/bb451/logAD \
	-v /etc/passwd:/etc/passwd \
	$image_name