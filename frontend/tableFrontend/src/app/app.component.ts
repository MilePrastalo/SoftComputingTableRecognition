import { Component } from '@angular/core';
import { ImageService } from './image.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'tableFrontend';
  image = '';
  noise = 0;

  constructor(private imageService: ImageService) { }


  openInput() {
    document.getElementById('fileInput').click();
  }

  fileChange(files) {
    const file = files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      this.image = reader.result.toString();
    };
    reader.onerror = error => {
      console.log('Error: ', error);
    };
  }
  sendImage() {
    this.imageService.uploadImage(this.image, this.noise).subscribe(
      (response => {
        console.log('success');
      }),
      (error => {
        console.log('error');
      })
    );
  }
}
