import { Injectable } from '@angular/core';
import { Observable } from 'rxjs/Observable';
import { TableData } from 'src/model/tableData';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ImageService {
  readonly serverLocation = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  uploadImage(image: string, noiseNum: number): Observable<TableData> {
    image = image.split(',')[1];
    console.log(image);
    console.log(noiseNum);
    let o = {
      pictureData: image,
      noise: noiseNum
    };
    return this.http.post<TableData>(this.serverLocation + '/image', o);
  }
}
