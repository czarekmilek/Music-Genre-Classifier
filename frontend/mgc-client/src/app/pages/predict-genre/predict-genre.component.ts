import { Component, signal } from '@angular/core';
import { FileUploadComponent } from '../../components/file-upload/file-upload.component';
import { ProbObject } from '../../models/predict-genre-response.model';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';
import { PredictGenreService } from '../../services/predict-genre.service';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { Router } from '@angular/router';
import { ProbObjectService } from '../../services/prob-object.service';

@Component({
  selector: 'app-predict-genre',
  imports: [FileUploadComponent, MatProgressSpinnerModule],
  templateUrl: './predict-genre.component.html',
  styleUrl: './predict-genre.component.scss',
})
export class PredictGenreComponent {
  constructor(
    private predictGenreService: PredictGenreService,
    private probObjectService: ProbObjectService,
    private router: Router
  ) {}
  isLoading = signal<boolean>(false);
  uploadProgress = 0;
  private subscriptions: Subscription = new Subscription();
  probObject: ProbObject | null = null;

  cancelAnalysis() {
    this.subscriptions.unsubscribe();
    this.subscriptions = new Subscription();
    this.isLoading.set(false);
    this.uploadProgress = 0;
  }
  handleFileUpload(file: File) {
    this.isLoading.set(true);
    const uploadSub = this.predictGenreService.uploadFile(file).subscribe({
      next: (event: any) => {
        if (event.type === HttpEventType.UploadProgress) {
          this.uploadProgress = Math.round((100 * event.loaded) / event.total);
        } else if (event instanceof HttpResponse) {
          const response = event.body as ProbObject;
          this.probObject = response;
          this.uploadProgress = 0;
          this.probObjectService.setProbObject(this.probObject);
          this.router.navigate(['/result-view']);
          this.isLoading.set(false);
        }
      },
      error: (error) => {
        console.error('Error while uploading the file:', error);
        alert('Failed to upload the file.');
        this.uploadProgress = 0;
        this.isLoading.set(false);
      },
    });
    this.subscriptions.add(uploadSub);
  }
  ngOnDestroy() {
    this.subscriptions.unsubscribe();
  }
}
