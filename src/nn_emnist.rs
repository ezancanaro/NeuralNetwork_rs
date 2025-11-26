/**
 *  Copyright 2025 Eric Zancanaro
 *    
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
//http://neuralnetworksanddeeplearning.com/chap2.html
//https://www.3blue1brown.com/lessons/backpropagation-calculus#title
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};
/**
 * Web Archive do formato usado no dataset MNIST
 *
 * https://web.archive.org/web/20020622183530/http://yann.lecun.com/exdb/mnist/
 */

#[derive(Debug)]
pub struct ImageFileHeader {
    magic_number: u32,
    num_images: u32,
    rows: u32,
    cols: u32,
}
#[derive(Debug)]
pub struct LabelFileHeader {
    magic_number: u32,
    num_labels: u32,
}

pub struct Parser {
    label_file: std::fs::File,
    image_file: std::fs::File,
    cur_index: u64,
    labels_header: LabelFileHeader,
    images_header: ImageFileHeader,
}

impl LabelFileHeader {
    pub fn parse_header(file_data: &[u8]) -> LabelFileHeader {
        LabelFileHeader {
            magic_number: u32::from_be_bytes(file_data[0..4].try_into().unwrap()),
            num_labels: u32::from_be_bytes(file_data[4..8].try_into().unwrap()),
        }
    }
}
impl ImageFileHeader {
    pub fn parse_header(file_data: &[u8]) -> ImageFileHeader {
        ImageFileHeader {
            magic_number: u32::from_be_bytes(file_data[0..4].try_into().unwrap()),
            num_images: u32::from_be_bytes(file_data[4..8].try_into().unwrap()),
            rows: u32::from_be_bytes(file_data[8..12].try_into().unwrap()),
            cols: u32::from_be_bytes(file_data[12..16].try_into().unwrap()),
        }
    }
}

impl Parser {
    const LABEL_OFFSET: u8 = 8;
    const IMAGE_OFFSET: u8 = 16;
    const IMAGE_SIZE: u64 = 28 * 28;

    fn read_label_header(label_file: &mut std::fs::File) -> LabelFileHeader {
        let mut buffer: [u8; 64] = [0; 64];
        let bytes_read = label_file.read(&mut buffer);
        LabelFileHeader::parse_header(&buffer)
    }
    fn read_image_header(image_file: &mut std::fs::File) -> ImageFileHeader {
        let mut header_buffer: [u8; 32] = [0; 32];
        let bytes_read = image_file.read(&mut header_buffer);
        ImageFileHeader::parse_header(&header_buffer)
    }

    pub fn setup(label_file_name: &str, image_file_name: &str) -> Parser {
        let mut label_file = std::fs::File::open(label_file_name).unwrap();
        let mut image_file = std::fs::File::open(image_file_name).unwrap();
        Parser {
            cur_index: 0,
            images_header: Parser::read_image_header(&mut image_file),
            labels_header: Parser::read_label_header(&mut label_file),
            label_file: label_file,
            image_file: image_file,
        }
    }

    pub fn transpose(image_buffer: &[u8], transposed_buffer: &mut[u8]){
        for row in 0..28 {
            for col in 0..28 {
                // Lendo do EMNIST (Topo -> Base), transpondo linhas/colunas
                let pixel = image_buffer[col * 28 + row];
                // Escrevendo no BMP (Base -> Topo)
                // A linha 0 do BMP é a linha 27 da imagem original
                let bmp_row = 27 - row;
                let bmp_index = bmp_row * 28 + col;
                transposed_buffer[bmp_index] = pixel;
            }
        }
    }

    pub fn read_next(&mut self) -> (Vec<u8>, u8) {
        let next_label_offset = Parser::LABEL_OFFSET as u64 + (self.cur_index);
        let next_image_offset = Parser::IMAGE_OFFSET as u64 + (Parser::IMAGE_SIZE * self.cur_index);

        self.label_file
            .seek(SeekFrom::Start((next_label_offset)))
            .unwrap();
        let mut label_buffer: [u8; 1] = [0];
        let label: Result<usize, std::io::Error> = self.label_file.read(&mut label_buffer);

        let mut image_buffer: [u8; 28 * 28] = [0; 28 * 28];
        let mut transposed_buffer: [u8; 28 * 28] = [0; 28 * 28];
        self.image_file
            .seek(SeekFrom::Start((next_image_offset)))
            .unwrap();
        let img = self.image_file.read(&mut image_buffer);

        Parser::transpose(&image_buffer, &mut transposed_buffer);

        self.cur_index += 1;

        (transposed_buffer.to_vec(), label_buffer[0])
    }

    pub fn has_more(&self)->bool{
        return self.cur_index < self.labels_header.num_labels as u64
    }

}

/**
 * Struct para gerar imagens no formato bmp.
 * Usada apenas para validar o parser visualmente
 */
pub struct Bitmap {
    signature: u16,
    size: u32,
    reserved: u32,
    offset: u32,
}
pub struct Dib {
    size: u32,
    width: u32,
    height: u32,
    planes: u16,
    bit_count: u16,
    compression: u32,
    image_size: u32,
    x_per_m: u32,
    y_per_m: u32,
    colors_used: u32,
    colors_important: u32,
}

impl Dib {
    pub fn new(w: u32, h: u32, bpp: u16, size: u32) -> Dib {
        Dib {
            size: 40,
            width: w,
            height: h,
            planes: 1,
            bit_count: bpp,
            compression: 0,
            image_size: size,
            x_per_m: 28,
            y_per_m: 28,
            colors_used: 256,
            colors_important: 0,
        }
    }

    pub fn to_vec(&self) -> [u8; 40] {
        let mut array: [u8; 40] = [0; 40];
        array[0..4].copy_from_slice(&self.size.to_le_bytes());
        array[4..8].copy_from_slice(&self.width.to_le_bytes());
        array[8..12].copy_from_slice(&self.height.to_le_bytes());
        array[12..14].copy_from_slice(&self.planes.to_le_bytes());
        array[14..16].copy_from_slice(&self.bit_count.to_le_bytes());
        array[16..20].copy_from_slice(&self.compression.to_le_bytes());
        array[20..24].copy_from_slice(&self.image_size.to_le_bytes());
        array[24..28].copy_from_slice(&self.x_per_m.to_le_bytes());
        array[28..32].copy_from_slice(&self.y_per_m.to_le_bytes());
        array[32..36].copy_from_slice(&self.colors_used.to_le_bytes());
        array[36..40].copy_from_slice(&self.colors_important.to_le_bytes());
        return array;
    }
}

impl Bitmap {
    pub fn new(data_len: u32) -> Bitmap {
        Bitmap {
            signature: 0x4D42,
            size: 40 + 14 + data_len,
            reserved: 0,
            offset: 54 + 1024,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn to_vec(&self) -> [u8; 14] {
        let mut array: [u8; 14] = [0; 14];
        array[0..2].copy_from_slice(&self.signature.to_le_bytes());
        array[2..6].copy_from_slice(&self.size.to_le_bytes());
        array[6..10].copy_from_slice(&self.reserved.to_le_bytes());
        array[10..14].copy_from_slice(&self.offset.to_le_bytes());
        return array;
    }

    pub fn create_bitmap(image_buffer: &[u8]) -> Vec<u8> {
        let dib = Dib::new(28, 28, 8, 0);
        let bitmap = Bitmap::new(image_buffer.len() as u32);
        let mut byte_array: [u8; 28 * 28 + 54 + 1024] = [0; 28 * 28 + 54 + 1024];
        byte_array[0..14].copy_from_slice(&bitmap.to_vec());
        byte_array[14..54].copy_from_slice(&dib.to_vec());

        for i in 0..256 {
            //collor pallete from 54 to 1078
            let idx = 54 + (i * 4);
            byte_array[idx] = i as u8; // Blue
            byte_array[idx + 1] = i as u8; // Green
            byte_array[idx + 2] = i as u8; // Red
            byte_array[idx + 3] = 0; // Reserved (Alpha/Padding)
        }
        let offset_pixels = 1078;
        byte_array[1078..].copy_from_slice(&image_buffer);
        byte_array.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{Seek, SeekFrom},
    };

    // Note this useful idiom: importing names from outer (for mod tests) scope. (Rust Book)
    use super::*;
    #[test]
    pub fn parse_labels() {
        let mut file = File::open("emnist/emnist-digits-train-labels-idx1-ubyte").unwrap();
        let mut buffer: [u8; 64] = [0; 64];

        let bytes_read = file.read(&mut buffer);

        let header = LabelFileHeader::parse_header(&buffer);

        println!(
            "Magic: {:?}, Num_Labels: {:?}",
            header.magic_number, header.num_labels
        );
        file.seek(SeekFrom::Start((8))).unwrap();
        let mut label_buffer: [u8; 1] = [0];
        let lb = file.read(&mut label_buffer);
        println!("Label: {:?}", label_buffer[0]);
    }

    #[test]
    pub fn parse_images() {
        let mut file = File::open("emnist/emnist-digits-train-images-idx3-ubyte").unwrap();

        // Images are 28x28, each pixel is 1 byte. Each Image is 784 bytes
        // File stores pixels row-wise
        let mut header_buffer: [u8; 32] = [0; 32];

        let bytes_read = file.read(&mut header_buffer);
        assert!(bytes_read.unwrap() == 32);

        let header = ImageFileHeader::parse_header(&header_buffer);

        println!(
            "Magic: {:?}, Num_Labels: {:?}, Rows:{:?}, Cols: {:?}",
            header.magic_number, header.num_images, header.rows, header.cols
        );

        let mut image_buffer: [u8; 28 * 28] = [0; 28 * 28];
        file.seek(SeekFrom::Start((16))).unwrap();

        let mut i = 0;
        while let Ok(bytes_read) = file.read(&mut image_buffer) {
            if bytes_read == 0 || i > 5 {
                // End of file reached
                break;
            }
            i += 1;

            let byte_array = Bitmap::create_bitmap(&image_buffer);
            // let inverted = &mut image_buffer[..];
            // inverted.reverse();
            // byte_array[1078..].copy_from_slice(&inverted);
            println!("Header: {:x} {:x}", byte_array[0], byte_array[1]);

            let write = std::fs::write(format!("bitmap_digit{}.bmp", i), byte_array);
            assert!(write.is_ok());
        }
    }

    #[test]
    pub fn test_parser() {
        let mut parser = Parser::setup(
            "emnist/emnist-digits-train-labels-idx1-ubyte",
            "emnist/emnist-digits-train-images-idx3-ubyte",
        );

        for i in 1..5 {
            let (img, label) = parser.read_next();
            println!("Label: {}", label);
            let bitmap = Bitmap::create_bitmap(&img);
            let write = std::fs::write(format!("parsed_digit{}.bmp", i), bitmap);
        }
    }
}
