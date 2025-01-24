import { Component } from '@angular/core';
import {FileUploadComponent} from '../../components/file-upload/file-upload.component';

@Component({
  selector: 'app-predict-genre',
  imports: [
    FileUploadComponent
  ],
  templateUrl: './predict-genre.component.html',
  styleUrl: './predict-genre.component.scss'
})
export class PredictGenreComponent {

}
