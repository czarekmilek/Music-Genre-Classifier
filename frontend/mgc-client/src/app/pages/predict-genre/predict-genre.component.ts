import { Component } from '@angular/core';
import { FileUploadComponent } from '../../components/file-upload/file-upload.component';
import { ProbObject } from '../../models/predict-genre-response.model';

@Component({
  selector: 'app-predict-genre',
  imports: [FileUploadComponent],
  templateUrl: './predict-genre.component.html',
  styleUrl: './predict-genre.component.scss',
})
export class PredictGenreComponent {
  handleClassificationResult(result: ProbObject) {
    console.log('Received classification result:', result);
  }
}
