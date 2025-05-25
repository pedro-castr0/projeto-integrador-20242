byte[] data = loadBytes("npy/tree.npy");

//int total = (data.length - 80) / 784;
//println(total);
int total = 100000;

byte[] outdata = new byte[total*784];
int outindex = 0;

for (int n = 0; n < total; n++){
  int start = 80 + n * 784;
  //PImage img = createImage(28, 28, RGB);
  //img.loadPixels();
  
  for (int i = 0; i < 784; i++){
    int index = i + start;
    byte val = data[index];
    outdata[outindex] = val;
    outindex++;
    //img.pixels[i] = color(val & 0xFF);
  }
  
  //img.updatePixels();
  //int x = 28 * (n % 10);
  //int y = 28 * (n / 10);
  //image(img,x,y);
}

saveBytes("bin/tree.bin", outdata);
