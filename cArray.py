def bin_to_c_array(input_file, output_file, array_name="model_tflite"):
    with open(input_file, "rb") as f:
        data = f.read()

    with open(output_file, "w") as f:
        f.write(f"// Converted from {input_file} to C array\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 12 >= len(data):  # last line, remove trailing comma
                line = line.rstrip(",")
            f.write(f"  {line},\n")

        f.write("};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")
        print(f"C array written to '{output_file}' with {len(data)} bytes.")

# Run it
bin_to_c_array("logistic_model.tflite", "logistic_model.h")
